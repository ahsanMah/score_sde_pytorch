# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
from tensorflow_datasets.core import dataset_info
import jax
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_addons as tfa


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(
        image, [h, w], antialias=True, method=tf.image.ResizeMethod.BICUBIC
    )


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False, ood_eval=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = (
        config.training.batch_size if not evaluation else config.eval.batch_size
    )
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch sizes ({batch_size} must be divided by"
            f"the number of devices ({jax.device_count()})"
        )

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10  # 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    # Create dataset builders for each dataset.
    if config.data.dataset == "CIFAR10":
        dataset_builder = tfds.builder("cifar10")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "SVHN":
        dataset_builder = tfds.builder("svhn_cropped")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(
                img, [config.data.image_size, config.data.image_size], antialias=True
            )

    elif config.data.dataset == "CELEBA":
        dataset_builder = tfds.builder("celeb_a")
        train_split_name = "train"
        eval_split_name = "validation"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = central_crop(img, 140)
            img = resize_small(img, config.data.image_size)
            return img

    elif config.data.dataset == "LSUN":
        dataset_builder = tfds.builder(f"lsun/{config.data.category}")
        train_split_name = "train"
        eval_split_name = "validation"

        if config.data.image_size == 128:

            def resize_op(img):
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = resize_small(img, config.data.image_size)
                img = central_crop(img, config.data.image_size)
                return img

        else:

            def resize_op(img):
                img = crop_resize(img, config.data.image_size)
                img = tf.image.convert_image_dtype(img, tf.float32)
                return img

    elif config.data.dataset in ["FFHQ", "CelebAHQ"]:
        dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
        train_split_name = "train"
        eval_split_name = "val"

    elif config.data.dataset == "MVTEC":
        dataset_dir = f"{config.data.dir_path}/{config.data.category}"
        dataset_builder = tfds.ImageFolder(dataset_dir)

        img_sz = config.data.downsample_size

        # Downsample to image size directly as no cropping will be done
        if evaluation:
            img_sz = config.data.image_size

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(
                img,
                [img_sz, img_sz],
                antialias=True,
                method=tf.image.ResizeMethod.LANCZOS5,
            )
            img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
            return img

        def augment_op(img):

            crop_sz = config.data.image_size
            img_sz = config.data.downsample_size

            # Random translate + rotate
            translate_ratio = 0.5 * (crop_sz / img_sz)
            img = tfa.image.rotate(img, tf.random.uniform((1,), 0, np.pi / 2))
            img = tfa.image.translate(
                img,
                tf.random.uniform(
                    (1, 2), -translate_ratio * img_sz, translate_ratio * img_sz
                ),
            )
            img = tf.image.resize_with_crop_or_pad(img, crop_sz, crop_sz)
            img = tf.image.random_hue(img, max_delta=0.05)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_flip_up_down(img)

            return img

        train_split_name = eval_split_name = "train"

        if ood_eval:
            train_split_name = "inlier"
            eval_split_name = "ood"

    elif config.data.dataset == "knee":
        dataset_dir = f"{config.data.dir_path}/{config.data.category}"
        dataset_builder = tfds.ImageFolder(dataset_dir)

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(
                img,
                [config.data.downsample_size, config.data.downsample_size],
                antialias=True,
                method=tf.image.ResizeMethod.LANCZOS5,
            )
            img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
            return img

        def augment_op(img):

            crop_sz = config.data.image_size
            img_sz = config.data.downsample_size

            # Random translate + rotate
            translate_ratio = 0.5 * (crop_sz / img_sz)
            img = tfa.image.rotate(img, tf.random.uniform((1,), 0, np.pi / 2))
            img = tfa.image.translate(
                img,
                tf.random.uniform(
                    (1, 2), -translate_ratio * img_sz, translate_ratio * img_sz
                ),
            )
            img = tf.image.resize_with_crop_or_pad(img, crop_sz, crop_sz)
            img = tf.image.random_hue(img, max_delta=0.05)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_flip_up_down(img)

            return img

    else:
        raise NotImplementedError(f"Dataset {config.data.dataset} not yet supported.")

    # Customize preprocess functions for each dataset.
    if config.data.dataset in ["FFHQ", "CelebAHQ"]:

        def preprocess_fn(d):
            sample = tf.io.parse_single_example(
                d,
                features={
                    "shape": tf.io.FixedLenFeature([3], tf.int64),
                    "data": tf.io.FixedLenFeature([], tf.string),
                },
            )
            data = tf.io.decode_raw(sample["data"], tf.uint8)
            data = tf.reshape(data, sample["shape"])
            data = tf.transpose(data, (1, 2, 0))
            img = tf.image.convert_image_dtype(data, tf.float32)
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0
            return dict(image=img, label=None)

    else:

        def preprocess_fn(d):
            """Basic preprocessing function scales data to [0, 1) and randomly flips."""
            img = resize_op(d["image"])
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0

            if config.data.dataset == "MVTEC" and not evaluation:
                img = augment_op(img)

            return dict(image=img, label=d.get("label", None))

    def create_dataset(dataset_builder, split, val=False):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)

        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            if not config.data.dataset == "MVTEC":
                dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
                split=split, shuffle_files=evaluation, read_config=read_config
            )
        else:
            ds = dataset_builder.with_options(dataset_options)

        if config.data.dataset == "MVTEC" and not ood_eval:
            val_size = int(0.1 * dataset_builder.info.splits["train"].num_examples)
            if val:
                ds = ds.take(val_size)
            else:  # train split
                ds = ds.skip(val_size)

        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.batch(batch_size, drop_remainder=False)
        return ds.prefetch(prefetch_size)

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name, val=True)

    # Test if loader worked
    # import matplotlib.pyplot as plt

    # for x in train_ds:
    #     print("Shape:", x["image"].shape)
    #     print(x["image"].numpy().max())
    #     plt.imshow(x["image"][0])
    #     plt.savefig("mvtec_test.png")
    #     break
    # exit()

    return train_ds, eval_ds, dataset_builder
