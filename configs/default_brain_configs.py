import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 32
    training.n_iters = 1000001
    training.snapshot_freq = 10001
    training.log_freq = 50
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 1000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.enable_augs = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.075

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 3
    evaluate.end_ckpt = 3
    evaluate.batch_size = 128
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = True
    evaluate.bpd_dataset = "inlier"
    evaluate.ood_eval = True

    # msma
    config.msma = msma = ml_collections.ConfigDict()
    msma.min_sigma = 1e-2  # Ignore first 10% of sigmas
    msma.n_timesteps = 20  # Number of discrete timesteps to evaluate
    msma.checkpoint = -1
    msma.zscale = False
    msma.ignore_ds = "train"

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "BRAIN"
    data.ood_ds = "LESION-c150"
    data.gen_ood = False
    data.uniform_dequantization = False
    data.centered = False
    data.num_channels = 2
    data.dir_path = "/DATA/Users/amahmood/braintyp/processed_v2/"
    data.splits_path = "/ahsan_projects/braintypicality/dataset/"
    data.cache_rate = 1.0
    data.image_size = 128
    data.mask_marginals = False
    data.select_channel = -1  # -1 = all, o/w indexed from zero
    data.dry_run = False
    data.spacing_pix_dim = 0.8
    data.workers = 8
    data.ood_ds_channel = 0

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 700  # TODO: Do this for brain ds!
    model.sigma_min = 0.01
    model.num_scales = 2000
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.0
    model.embedding_type = "fourier"
    model.amp = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    # DGMM params
    config.dgmm = dgmm = ml_collections.ConfigDict()
    dgmm.resume = False

    config.seed = 42
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    return config
