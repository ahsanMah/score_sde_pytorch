import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import tensorflow.keras as tfk

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision

from datetime import datetime
from tqdm.auto import tqdm
from numba import njit

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers
# tf.enable_v2_behavior()
tf.data.AUTOTUNE = tf.data.experimental.AUTOTUNE

EPSILON = 1e-5

loss_tracker = tf.keras.metrics.Mean(name="loss")


class DGMM(tf.keras.Model):
    """
    Deep GMM

    D : # Dimension of Multivariate Normals aka Event shape
    """

    def __init__(self, mask_shape, latent_dim=256, k_mixt=5, D=20):
        super(DGMM, self).__init__()

        self.resnet = tfk.applications.MobileNetV3Small(
            input_shape=(mask_shape[0], mask_shape[1], 1),
            alpha=1.0,
            include_top=False,
            weights=None,
            pooling="avg",
            minimalistic=True,
        )

        self.hidden = tfk.Sequential(
            [
                # tfk.layers.InputLayer(input_shape=mask_shape),
                # tfk.layers.Conv2D(filters=3, kernel_size=1),
                self.resnet,
                tfk.layers.Dropout(0.2),
            ],
            name="latent",
        )

        self.alpha = tfk.Sequential(
            [
                self.hidden,
                Dense(k_mixt, activation=None, kernel_initializer="he_uniform"),
                tfk.layers.Activation("linear", dtype="float32"),
            ],
            name="alpha",
        )

        self.mu = tfk.Sequential(
            [
                self.hidden,
                Dense(
                    k_mixt * D,
                    activation=tf.nn.elu,
                    name="mu",
                    kernel_initializer="he_uniform",
                ),
                Reshape((k_mixt, D), dtype="float32"),
            ],
            name="mu",
        )

        self.sigma = tfk.Sequential(
            [
                self.hidden,
                Dense(
                    k_mixt * (D * (D + 1) // 2),
                    activation=tf.nn.softplus,
                    # kernel_regularizer=tfk.regularizers.l2(1e-5),
                ),
                Reshape((k_mixt, (D * (D + 1)) // 2)),
                tfk.layers.Lambda(self.stable_lower_triangle, dtype="float32"),
            ],
            name="sigma",
        )

    @tf.function(experimental_compile=True)
    def stable_lower_triangle(self, x):
        eps = 1e-6
        x = tfp.math.fill_triangular(x + eps)
        return x

    @tf.function
    def log_pdf_univariate(self, x, y):

        gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.alpha(y)),
            components_distribution=tfd.Normal(loc=self.mu(y), scale=self.sigma(y)),
        )

        return gmm.log_prob(tf.reshape(x, (-1,)))

    @tf.function(experimental_compile=True)
    def log_loss(self, _, log_prob):
        return -tf.reduce_mean(log_prob)

    @tf.function(experimental_compile=True)
    def log_pdf(self, x, y):
        gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.alpha(y)),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=self.mu(y), scale_tril=self.sigma(y)
            ),
        )
        return gmm.log_prob(x)

    @tf.function(experimental_compile=True, experimental_relax_shapes=True)
    def call(self, inputs):
        score, mask = inputs
        # print(x.dtype, y.dtype)
        log_prob = self.log_pdf(
            tf.cast(score, dtype=tf.float32), tf.cast(mask, dtype=tf.float32)
        )
        return log_prob

    def build_likelihood_fn(self, mask):

        gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=self.alpha(mask)),
            components_distribution=tfd.MultivariateNormalTriL(
                loc=self.mu(mask), scale_tril=self.sigma(mask)
            ),
        )

        @tf.function(experimental_compile=True)
        def ll_fn(x):
            return gmm.log_prob(x)

        return ll_fn


@tf.function
def gmm_train_step(model, opt, batch):
    with tf.GradientTape() as t:
        log_lls = model(batch)
        current_loss = model.log_loss(None, log_lls)

    gradients = t.gradient(current_loss, model.trainable_variables)
    opt_op = opt.apply_gradients(zip(gradients, model.trainable_variables))

    return current_loss


def build_optimizer():
    return tf.optimizers.Adam(learning_rate=3e-4)


# def build_scorer(ckpt_fiilename, n_timesteps=20):


#     config = configs.get_config()
#     sde = subVPSDE(
#         beta_min=config.model.beta_min,
#         beta_max=config.model.beta_max,
#         N=config.model.num_scales,
#     )

#     sigmas = mutils.get_sigmas(config)
#     scaler = datasets.get_data_scaler(config)
#     inverse_scaler = datasets.get_data_inverse_scaler(config)
#     score_model = mutils.create_model(config)

#     optimizer = get_optimizer(config, score_model.parameters())
#     ema = ExponentialMovingAverage(
#         score_model.parameters(), decay=config.model.ema_rate
#     )
#     state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

#     state = restore_checkpoint(ckpt_filename, state, config.device)
#     ema.copy_to(score_model.parameters())
#     ckpt = state["step"]
#     print(f"Loaded checkpoint {ckpt}")
#     score_fn = mutils.get_score_fn(
#         sde, score_model, train=False, continuous=config.training.continuous
#     )
#     eps = config.msma.min_sigma
#     msma_sigmas = torch.linspace(eps, 1.0, n_timesteps, device="cuda")
#     timesteps = sde.noise_schedule_inverse(msma_sigmas)

#     def vectorized_score_norm_fn(x):
#         with torch.no_grad():
#             batch_sz = x.shape[0]
#             x = x.repeat_interleave(n_timesteps, dim=0)
#             vec_t = timesteps.repeat(batch_sz)
#             score = score_fn(x, vec_t)
#             score = score.view(batch_sz, n_timesteps, -1)
#             scores = torch.linalg.norm(score, axis=2) * msma_sigmas
#         return scores


"""
Starts a fresh round of training for `n_epochs`
"""


def trainer(model, optimizer, train_ds, val_ds, dataset, r_sz, n_samples, n_epochs=20):

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # an absolute change of less than min_delta, will count as no improvement
            min_delta=1e-3,
            # "no longer improving" being defined as "for at least patiencef epochs"
            patience=10,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath=f"saved_models/dgmm/{dataset}/{r_sz}x{r_sz}/" + "e{epoch}",
            # Only save a model if `val_loss` has improved.
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, min_delta=1e-3, patience=2, min_lr=1e-5
        ),
        tf.keras.callbacks.TensorBoard(
            f"./logs/dgmm/{dataset}/{r_sz}x{r_sz}_{start_time}", update_freq=1
        ),
    ]

    model.compile(optimizer=optimizer, loss=DGMM.log_loss)
    # model.load_weights("saved_models/dgmm_init")

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=n_epochs, callbacks=callbacks
    )

    # val_ds = test_ds.take(8).cache()

    # avg_loss = tfk.metrics.Mean()
    # val_loss = tfk.metrics.Mean()

    # n_steps = n_epochs * n_samples
    # epochs_bar = tqdm(range(n_epochs), desc="Epoch")
    # losses = [0]
    # val_losses = [0]

    # for i in epochs_bar:
    #   progress_bar = tqdm(train_ds, desc=f"Loss: {losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}", leave=False)
    #   for j, x in enumerate(progress_bar):
    #     loss = train_step(model, x["score"], x["image"], optimizer)
    #     avg_loss(loss)

    #     if j % 10 == 0:
    #       for x_val in val_ds:
    #         val_loss(log_loss(model, x_val["score"], x_val["image"]))

    #       losses.append(avg_loss.result())
    #       val_losses.append(val_loss.result())

    #       val_loss.reset_states()
    #       avg_loss.reset_states()

    #       progress_bar.set_description(f"Loss: {losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    #   model.save_weights(f"saved_models/dgmm/{dataset}/{r_sz}x{r_sz}/e{i}")

    return history
