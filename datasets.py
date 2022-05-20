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
import os, re
import glob
import jax
import torch
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import pandas as pd

from mri_utils import complex_magnitude, RandTumor
from fastmri import FastKnee, FastKneeTumor
from monai.data import CacheDataset, DataLoader, ArrayDataset, PersistentDataset
from monai.transforms import *

from torchvision import transforms
from PIL import Image

from DataLoaderAnomaly import AnomalyDataLoader, TransformFrames
import matplotlib.pyplot as plt
import seaborn as sns


class RandomPattern:
    """
    Reproduces "random pattern mask" for inpainting, which was proposed in
    Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T.,
    & Efros, A. A. Context Encoders: Feature Learning by Inpainting.
    Conference on Computer Vision and Pattern Recognition, 2016.
    ArXiv link: https://arxiv.org/abs/1604.07379
    This code is based on lines 273-283 and 316-330 of Context Encoders
    implementation:
    https://github.com/pathak22/context-encoder/blob/master/train_random.lua
    The idea is to generate small matrix with uniform random elements,
    then resize it using bicubic interpolation into a larger matrix,
    then binarize it with some threshold,
    and then crop a rectangle from random position and return it as a mask.
    If the rectangle contains too many or too few ones, the position of
    the rectangle is generated again.
    The big matrix is resampled when the total number of elements in
    the returned masks times update_freq is more than the number of elements
    in the big mask. This is done in order to find balance between generating
    the big matrix for each mask (which is involves a lot of unnecessary
    computations) and generating one big matrix at the start of the training
    process and then sampling masks from it only (which may lead to
    overfitting to the specific patterns).
    """

    def __init__(
        self, max_size=10000, resolution=0.06, density=0.25, update_freq=1, seed=239
    ):
        """
        Args:
            max_size (int):      the size of big binary matrix
            resolution (float):  the ratio of the small matrix size to
                                 the big one. Authors recommend to use values
                                 from 0.01 to 0.1.
            density (float):     the binarization threshold, also equals
                                 the average ones ratio in the mask
            update_freq (float): the frequency of the big matrix resampling
            seed (int):          random seed
        """
        self.max_size = max_size
        self.resolution = resolution
        self.density = density
        self.update_freq = update_freq
        self.rng = np.random.RandomState(seed)
        self.regenerate_cache()

    def regenerate_cache(self):
        """
        Resamples the big matrix and resets the counter of the total
        number of elements in the returned masks.
        """
        low_size = int(self.resolution * self.max_size)
        low_pattern = self.rng.uniform(0, 1, size=(low_size, low_size))
        low_pattern = low_pattern.astype("float32")
        pattern = Image.fromarray(low_pattern)
        pattern = pattern.resize((self.max_size, self.max_size), Image.BICUBIC)
        pattern = np.array(pattern)
        pattern = (pattern < self.density).astype("float32")
        self.pattern = pattern
        self.points_used = 0

    def __call__(self, image, density_std=0.05, channel_last=True):
        """
        Image is supposed to have shape [H, W, C].
        Return binary mask of the same shape, where for each object
        the ratio of ones in the mask is in the open interval
        (self.density - density_std, self.density + density_std).
        The less is density_std, the longer is mask generation time.
        For very small density_std it may be even infinity, because
        there is no rectangle in the big matrix which fulfills
        the requirements.
        """
        height, width = image.shape
        x = self.rng.randint(0, self.max_size - width + 1)
        y = self.rng.randint(0, self.max_size - height + 1)
        res = self.pattern[y : y + height, x : x + width]
        coverage = res.mean()
        while not (self.density - density_std < coverage < self.density + density_std):
            x = self.rng.randint(0, self.max_size - width + 1)
            y = self.rng.randint(0, self.max_size - height + 1)
            res = self.pattern[y : y + height, x : x + width]
            coverage = res.mean()
        # mask = np.tile(res[:, :, None], [1, 1, num_channels])
        if channel_last:
            mask = res[:, :, None]
        else:
            mask = res[None, :, :]

        mask = 1.0 - mask
        # sns.heatmap(mask[0])
        # plt.savefig(f"imgs/MASK.png", dpi=200)
        # exit()
        self.points_used += width * height
        if self.update_freq * (self.max_size ** 2) < self.points_used:
            self.regenerate_cache()
        return mask.astype("uint8")


# class RandomPattern:
#     """
#     Reproduces "random pattern mask" for inpainting, which was proposed in
#     Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T.,
#     & Efros, A. A. Context Encoders: Feature Learning by Inpainting.
#     Conference on Computer Vision and Pattern Recognition, 2016.
#     ArXiv link: https://arxiv.org/abs/1604.07379
#     This code is based on lines 273-283 and 316-330 of Context Encoders
#     implementation:
#     https://github.com/pathak22/context-encoder/blob/master/train_random.lua
#     The idea is to generate small matrix with uniform random elements,
#     then resize it using bicubic interpolation into a larger matrix,
#     then binarize it with some threshold,
#     and then crop a rectangle from random position and return it as a mask.
#     If the rectangle contains too many or too few ones, the position of
#     the rectangle is generated again.
#     The big matrix is resampled when the total number of elements in
#     the returned masks times update_freq is more than the number of elements
#     in the big mask. This is done in order to find balance between generating
#     the big matrix for each mask (which is involves a lot of unnecessary
#     computations) and generating one big matrix at the start of the training
#     process and then sampling masks from it only (which may lead to
#     overfitting to the specific patterns).
#     """

#     def __init__(
#         self, max_size=10000, resolution=0.06, density=0.25, update_freq=1, seed=239
#     ):
#         """
#         Args:
#             max_size (int):      the size of big binary matrix
#             resolution (float):  the ratio of the small matrix size to
#                                  the big one. Authors recommend to use values
#                                  from 0.01 to 0.1.
#             density (float):     the binarization threshold, also equals
#                                  the average ones ratio in the mask
#             update_freq (float): the frequency of the big matrix resampling
#             seed (int):          random seed
#         """
#         self.max_size = max_size
#         self.resolution = resolution
#         self.density = density
#         self.update_freq = update_freq
#         self.rng = np.random.RandomState(seed)
#         self.regenerate_cache()

#     def regenerate_cache(self):
#         """
#         Resamples the big matrix and resets the counter of the total
#         number of elements in the returned masks.
#         """
#         low_size = int(self.resolution * self.max_size)
#         low_pattern = self.rng.uniform(0, 1, size=(low_size, low_size)) * 255
#         low_pattern = torch.from_numpy(low_pattern.astype("float32"))
#         pattern = transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.Resize(self.max_size, Image.BICUBIC),
#                 transforms.ToTensor(),
#             ]
#         )(low_pattern[None])[0]
#         pattern = torch.lt(pattern, self.density).byte()
#         self.pattern = pattern.byte()
#         self.points_used = 0

#     def __call__(self, batch, density_std=0.05):
#         """
#         Batch is supposed to have shape [num_objects x num_channels x
#         x width x height].
#         Return binary mask of the same shape, where for each object
#         the ratio of ones in the mask is in the open interval
#         (self.density - density_std, self.density + density_std).
#         The less is density_std, the longer is mask generation time.
#         For very small density_std it may be even infinity, because
#         there is no rectangle in the big matrix which fulfills
#         the requirements.
#         """
#         batch_size, num_channels, width, height = batch.shape
#         res = torch.zeros_like(batch, device="cpu")
#         idx = list(range(batch_size))
#         while idx:
#             nw_idx = []
#             x = self.rng.randint(0, self.max_size - width + 1, size=len(idx))
#             y = self.rng.randint(0, self.max_size - height + 1, size=len(idx))
#             for i, lx, ly in zip(idx, x, y):
#                 res[i] = self.pattern[lx : lx + width, ly : ly + height][None]
#                 coverage = float(res[i, 0].mean())
#                 if not (
#                     self.density - density_std < coverage < self.density + density_std
#                 ):
#                     nw_idx.append(i)
#             idx = nw_idx
#         self.points_used += batch_size * width * height
#         if self.update_freq * (self.max_size ** 2) < self.points_used:
#             self.regenerate_cache()
#         return res


def get_channel_selector(config):
    c = config.data.select_channel
    if c > -1:
        return lambda x: np.expand_dims(x[c, ...], axis=0)
    else:
        return lambda x: x


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
    shuffle_buffer_size = 100  # 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    # TODO: Add appropriate OOD evaluation sets for other datasets
    if ood_eval and config.data.dataset in [
        "CIFAR10",
        "SVHN",
        "CELEBA",
        "LSUN",
        "FFHQ",
        "CelebAHQ",
    ]:
        raise NotImplementedError(
            f"OOD evaluation for dataset {config.data.dataset} not yet supported."
        )

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

        # This is image size BEFORE translate + crop
        # e.g 1024 -> 200 - augment -> crop to 128
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
            translate_ratio = 0.33 * (crop_sz / img_sz)
            img = tfa.image.rotate(img, tf.random.uniform((1,), 0, np.pi / 2))
            img = tfa.image.translate(
                img,
                tf.random.uniform(
                    (1, 2), -translate_ratio * img_sz, translate_ratio * img_sz
                ),
            )
            img = tf.image.resize_with_crop_or_pad(img, crop_sz, crop_sz)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            img = tf.image.random_brightness(img, max_delta=0.05)
            img = tf.image.random_hue(img, max_delta=0.05)
            img = tf.image.random_flip_up_down(img)

            return img

        tmp = tf.zeros((config.data.image_size, config.data.image_size, 1))
        mask_generator = RandomPattern()

        def mask_op(img):

            if evaluation:
                mask = tf.ones_like(tmp)
                img = tf.concat([img, mask], axis=-1)
                return img

            if np.random.uniform([1]) < 0.5:
                mask = tf.ones_like(img)
            else:
                mask = tf.cast(mask_generator(tmp), dtype=tf.float32)
                # mask = tf.expand_dims(mask, axis=0)

            # mask = tf.repeat(mask, repeats=img.shape[0], axis=0)
            img = img * mask
            img = tf.concat([img, mask], axis=-1)

            return img

        train_split_name = eval_split_name = "train"

        if ood_eval:
            train_split_name = "train"
            eval_split_name = "test"

    elif config.data.dataset == "KNEE":

        rimg_h = rimg_w = config.data.downsample_size

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize_with_pad(
                img,
                rimg_h,
                rimg_h,
                antialias=True,
                method=tf.image.ResizeMethod.LANCZOS5,
            )

            # img = img[15:-15, :]  # Crop high freqs to get square image

            return img

        if config.longleaf:
            dataset_dir = f"{config.data.dir_path_longleaf}"
        else:
            dataset_dir = f"{config.data.dir_path}"

        max_marginal_ratio = config.data.marginal_ratio
        mask_marginals = config.data.mask_marginals
        category = config.data.category
        complex_input = config.data.complex

        train_dir = os.path.join(dataset_dir, "singlecoil_train/")
        val_dir = os.path.join(dataset_dir, "singlecoil_val/")
        test_dir = os.path.join(dataset_dir, "singlecoil_test_v2/")

        img_h, img_w = config.data.original_dimensions
        c = 2 if complex_input else 1

        # if config.mask_marginals:
        #     c += 1

        def normalize(img, complex_input=False, quantile=0.999):

            # Complex tensors are 2D
            if complex_input:
                h = np.quantile(img.reshape(-1, 2), q=quantile, axis=0)
                # l = np.min(img.reshape(-1, 2), axis=0)
                l = np.quantile(img.reshape(-1, 2), q=(1 - quantile) / 10, axis=0)
            else:
                h = np.quantile(img, q=quantile)
                # l = np.min(img)
                l = np.quantile(img, q=(1 - quantile) / 10)

            # Min Max normalize
            img = (img - l) / (h - l)
            img = np.clip(
                img,
                0.0,
                1.0,
            )

            return img

        def make_generator(ds, ood=False):

            if complex_input:
                # Treat input as a 3D tensor (2 channels: real + imag)
                preprocessor = lambda x: np.stack([x.real, x.imag], axis=-1)
                normalizer = lambda x: normalize(x, complex_input=True)
            else:
                preprocessor = lambda x: complex_magnitude(x).numpy()[..., np.newaxis]
                normalizer = lambda x: normalize(x)

            label = 1 if ood else 0

            # TODO: Build complex loader for img

            def tf_gen_img():
                for k, x in ds:
                    img = preprocessor(x)
                    img = normalizer(img)
                    yield img

            def tf_gen_ksp():
                for k, x in ds:
                    img = preprocessor(k)
                    img = normalizer(img)
                    yield img

            if "kspace" == category:
                print(
                    f"Training on {'complex' if complex_input else 'image'} kspace..."
                )
                return tf_gen_ksp

            # Default to target image as category
            print(f"Training on {'complex' if complex_input else 'image'} mri...")
            return tf_gen_img

        def build_ds(datadir, ood=False):

            output_type = tf.float32
            output_shape = tf.TensorShape([img_h, img_w, c])

            dataset = FastKnee(datadir) if not ood_eval else FastKneeTumor(datadir)
            ds = tf.data.Dataset.from_generator(
                make_generator(dataset, ood=ood),
                output_type,
                output_shape,
                # output_signature=(tf.TensorSpec(shape=(img_h, img_w, c), dtype=tf.float32)),
            )

            return ds

        channels = config.data.num_channels

        def np_build_and_apply_random_mask(x):
            # Building mask of random columns to **keep**
            # img_h, img_w, c = x.shape
            bs = x.shape[0]

            rand_ratio = np.random.uniform(
                low=config.data.min_marginal_ratio,
                high=config.data.marginal_ratio,
                size=1,
            )
            n_mask_cols = int(rand_ratio * rimg_w)
            rand_cols = np.random.randint(rimg_w, size=n_mask_cols)

            # We do *not* want to mask out the middle (low) frequencies
            # Keeping 10% of low freq is equivalent to Scenario-30L in activemri paper
            low_freq_cols = np.arange(int(0.45 * rimg_w), img_w - int(0.45 * rimg_w))
            mask = np.zeros((bs, rimg_h, rimg_w, 1), dtype=np.float32)
            mask[..., rand_cols, :] = 1.0
            mask[..., low_freq_cols, :] = 1.0

            # Applying + Appending mask
            x = x * mask
            x = np.concatenate([x, mask], axis=-1)
            return x

        test_slices = 2000
        train_ds = build_ds(train_dir)
        eval_ds = build_ds(val_dir).skip(test_slices)

        # The datsets used to evaluate MSMA
        if ood_eval:
            train_ds = build_ds(val_dir).take(test_slices)
            eval_ds = build_ds(val_dir, ood=True).take(test_slices)

        dataset_builder = train_split_name = eval_split_name = None

    elif config.data.dataset == "BRAIN":
        dataset_dir = config.data.dir_path
        splits_dir = config.data.splits_path

        clean = lambda x: x.strip().replace("_", "")
        # print("Dir for keys:", splits_dir)
        filenames = {}
        for split in ["train", "val", "test"]:
            with open(os.path.join(splits_dir, f"{split}_keys.txt"), "r") as f:
                filenames[split] = [clean(x) for x in f.readlines()]

        val_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
            for x in filenames["val"]
        ]

        train_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
            for x in filenames["train"]
        ]

        test_file_list = [
            {"image": os.path.join(dataset_dir, f"{x}.nii.gz")}
            for x in filenames["test"]
        ]

        img_sz = config.data.image_size
        cache = config.data.cache_rate
        channel_selector = get_channel_selector(config)
        cache_dir_name = "/tmp/monai_brains_nopad/train"

        tmp = torch.zeros((config.data.image_size, config.data.image_size))
        # print(tmp.shape)
        mask_generator = RandomPattern()

        def fetch_mid_axial_slices(x, val=False):
            if evaluation:
                fetched_slice = x.shape[3] // 2
            else:  # [mid-2 : mid+2]
                fetched_slice = np.random.randint(
                    x.shape[3] // 2 - 2, x.shape[3] // 2 + 3
                )

            return channel_selector(x[:, :, fetched_slice, :])

        def mask_op(img):

            if not config.data.mask_marginals:
                return img

            if evaluation:
                mask = torch.ones((1, img_sz, img_sz))
                img = torch.cat((img, mask), dim=0)
                return img

            if np.random.uniform([1]) < 0.5:
                mask = torch.ones((1, img_sz, img_sz))
            else:
                mask = torch.from_numpy(mask_generator(tmp, channel_last=False)).float()

            img = img * mask
            img = torch.cat((img, mask), dim=0)

            # sns.heatmap(img[0])
            # plt.savefig(f"imgs/BRAIN.png", dpi=200)
            # exit()
            return img

        loading_transforms = [
            LoadImaged("image", image_only=True),
            SqueezeDimd("image", dim=3),
            AsChannelFirstd("image"),
            SpatialCropd("image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]),
        ]

        augmentations = [
            RandStdShiftIntensityd("image", (-0.1, 0.1)),
            RandScaleIntensityd("image", (-0.1, 0.1)),
            RandHistogramShiftd("image", num_control_points=[3, 5]),
            RandFlipd("image", prob=0.5, spatial_axis=0),
            RandAffined(
                "image",
                prob=0.1,
                rotate_range=[0.03, 0.03, 0.03],
                translate_range=3,
            ),
        ]
        post_proc_transforms = [
            RandLambdad("image", func=fetch_mid_axial_slices),
            ToTensord("image"),
            TorchVisiond("image", "Resize", size=(img_sz, img_sz), antialias=True),
            ScaleIntensityd("image", minv=0, maxv=1.0),
            RandLambdad("image", func=mask_op),
        ]

        train_transform = list(loading_transforms)
        if config.training.enable_augs:
            train_transform.extend(augmentations)
        train_transform.extend(post_proc_transforms)

        train_transform = Compose(train_transform)

        val_transform = list(loading_transforms)
        val_transform.extend(post_proc_transforms)
        val_transform = Compose(val_transform)

        # train_ds = ArrayDataset(
        #     train_file_list, img_transform=train_transform
        # )

        if not evaluation:
            # train_ds = PersistentDataset(
            #     train_file_list,
            #     transform=train_transform,
            #     cache_dir=cache_dir_name,
            # )
            train_ds = CacheDataset(
                train_file_list,
                transform=train_transform,
                cache_rate=cache,
                num_workers=10,
            )
            eval_ds = CacheDataset(
                val_file_list, transform=val_transform, cache_rate=cache, num_workers=10
            )

        elif not ood_eval:
            train_ds = CacheDataset(
                train_file_list,
                transform=val_transform,
                cache_rate=cache,
                num_workers=6,
            )

            eval_ds = CacheDataset(
                val_file_list,
                transform=val_transform,
                cache_rate=cache,
                num_workers=6,
            )

        else:  # evaluation AND ood_eval
            train_ds = None
            inlier_file_list = test_file_list
            img_transform = val_transform

            # Generate OOD samples by adding "tumors" to center
            # i.e. compute random grid deformations
            if config.data.gen_ood:
                deformer = RandTumor(
                    spacing=1.0,
                    max_tumor_size=15.0 / config.data.spacing_pix_dim,
                    magnitude_range=(
                        5.0 / config.data.spacing_pix_dim,
                        15.0 / config.data.spacing_pix_dim,
                    ),
                    prob=1.0,
                    spatial_size=[168, 152],  # [168, 200, 152],
                    padding_mode="zeros",
                )

                deformer.set_random_state(seed=0)

                ood_transform = Compose(
                    [
                        LoadImaged("image", image_only=True),
                        SqueezeDimd("image", dim=3),
                        AsChannelFirstd("image"),
                        SpatialCropd(
                            "image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]
                        ),
                        RandLambdad("image", func=fetch_mid_axial_slices),
                        RandLambdad("image", deformer),
                        ToTensord("image"),
                        TorchVisiond(
                            "image", "Resize", size=(img_sz, img_sz), antialias=True
                        ),
                        ScaleIntensityd("image", minv=0, maxv=1.0),
                        RandLambdad("image", func=mask_op),
                    ]
                )

                ood_file_list = test_file_list
                img_transform = ood_transform

            elif config.data.ood_ds == "IBIS":
                filenames = {}
                for split in ["ibis_inlier", "ibis_outlier"]:
                    with open(os.path.join(splits_dir, f"{split}_keys.txt"), "r") as f:
                        filenames[split] = [x.strip() for x in f.readlines()]

                inlier_file_list = [
                    {"image": os.path.join(dataset_dir, "ibis", f"{x}.nii.gz")}
                    for x in filenames["ibis_inlier"]
                ]

                ood_file_list = [
                    {"image": os.path.join(dataset_dir, "ibis", f"{x}.nii.gz")}
                    for x in filenames["ibis_outlier"]
                ]

            elif "LESION" in config.data.ood_ds:

                # channel_selector = lambda x: np.expand_dims(x[0, ...], axis=0)
                # Selects the first channel and middle slice
                c = config.data.ood_ds_channel

                def fetch_lesion_slices(x):
                    fetched_slice = x.shape[3] // 2
                    return np.expand_dims(x[c, :, fetched_slice, :], axis=0)

                img_transform = Compose(
                    [
                        LoadImaged("image", image_only=True),
                        SqueezeDimd("image", dim=3),
                        AsChannelFirstd("image"),
                        SpatialCropd(
                            "image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]
                        ),
                        RandLambdad("image", func=fetch_lesion_slices),
                        ToTensord("image"),
                        TorchVisiond(
                            "image", "Resize", size=(img_sz, img_sz), antialias=True
                        ),
                        ScaleIntensityd("image", minv=0, maxv=1.0),
                        RandLambdad("image", func=mask_op),
                    ]
                )

                suffix = ""
                if "-" in config.data.ood_ds:
                    _, suffix = config.data.ood_ds.split("-")
                    suffix = "-" + suffix

                dirname = "lesion" + suffix

                skip_train_ids = (
                    lambda x: re.search("(NDAR.*).nii.gz", x).group(1)
                    not in filenames["train"]
                )

                ood_file_list = [
                    {"image": x}
                    for x in filter(
                        skip_train_ids,
                        glob.glob(os.path.join(dataset_dir, "..", dirname, "*")),
                    )
                ]
                print("Collected samples:", len(ood_file_list), "from", dirname)

            # Load either real or generated ood samples
            # Defaults to ABCD test/ood data
            train_ds = CacheDataset(
                inlier_file_list,
                transform=val_transform,
                cache_rate=0,
                num_workers=4,
            )

            eval_ds = CacheDataset(
                ood_file_list,
                transform=img_transform,
                cache_rate=0,
                num_workers=4,
            )

    elif config.data.dataset == "FAMLI":
        dataset_dir = "/FAMLI/Shared/C1_ML_Analysis/"
        train_df = pd.read_csv(
            os.path.join(
                dataset_dir,
                "CSV_files",
                "ALL_C1_C2_cines_gt_ga_withmeta_masked_resampled_256_spc075_uuid_study_flyto_uuid_train.csv",
            )
        )
        # print(np.unique(train_df.tag, return_counts=True))

        train_df = train_df.loc[train_df["tag"].isin(["BPD", "HC"])]
        train_df = train_df.loc[:, ["file_path", "ga_boe", "tag", "study_id"]]
        train_df = train_df.query("126 <= ga_boe and ga_boe <= 210").reset_index(
            drop=True
        )

        eval_df = pd.read_csv(
            os.path.join(
                dataset_dir,
                "CSV_files",
                "ALL_C1_C2_cines_gt_ga_withmeta_masked_resampled_256_spc075_uuid_study_flyto_uuid_valid.csv",
            )
        )
        # print(np.unique(val_df.tag, return_counts=True))

        INLIER_TAGS = ["BPD", "HC"]
        OOD_TAGS = (
            ["FL", "AC"] if config.data.ood_ds == "TAGS" else [config.data.ood_ds]
        )  # ["FL", "AC"]

        tags = OOD_TAGS if ood_eval else INLIER_TAGS
        print(f"Using tags: {tags}")

        val_df = eval_df.loc[eval_df["tag"].isin(INLIER_TAGS)]

        if ood_eval and not config.data.gen_ood:
            ood_df = eval_df.loc[eval_df["tag"].isin(OOD_TAGS)]
            # ood_df = ood_df.loc[~ood_df.study_id.isin(val_df.study_id)]
            # print("USING EXCLUSIVE OODs")
            val_df = ood_df

        val_df = val_df.loc[:, ["file_path", "ga_boe", "tag"]]
        val_df = val_df.query("126 <= ga_boe and ga_boe <= 210").reset_index(drop=True)

        print(f"Collected: {val_df.shape[0]} samples")

        train_ds = AnomalyDataLoader(
            train_df, mount_point=dataset_dir, transform=TransformFrames(training=True)
        )

        eval_ds = AnomalyDataLoader(
            val_df if not ood_eval else val_df[0:1000:5].reset_index(),
            mount_point=dataset_dir,
            transform=TransformFrames(training=False, ood=config.data.gen_ood),
            random_last=10,
        )

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

            if config.data.dataset in ["KNEE"]:
                d = {"image": d}

            img = resize_op(d["image"])
            if config.data.random_flip and not evaluation:
                img = tf.image.random_flip_left_right(img)
            if uniform_dequantization:
                img = (
                    tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
                ) / 256.0

            if config.data.dataset == "MVTEC":

                if not evaluation and config.training.rand_augment:
                    img = augment_op(img)

                if config.data.mask_marginals:
                    img = mask_op(img)

            # if config.data.dataset == "KNEE" and config.data.mask_marginals:
            #     img = np_build_and_apply_random_mask(img)

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
        elif dataset_builder not in ["KNEE"]:
            ds = dataset_builder.with_options(dataset_options)
        else:  # dataset_builder is already a TF Dataset
            ds = dataset_builder

        if config.data.dataset == "MVTEC" and not ood_eval:
            val_size = int(0.1 * dataset_builder.info.splits["train"].num_examples)
            if val:
                ds = ds.take(val_size)
            else:  # train split
                ds = ds.skip(val_size)

        ds = ds.cache()
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)

        if config.data.dataset == "KNEE" and config.data.mask_marginals:
            _fn = lambda x: tf.numpy_function(
                func=np_build_and_apply_random_mask, inp=[x], Tout=tf.float32
            )

            def mask_fn(d):
                x = d["image"]
                l = d["label"]

                return {"image": _fn(x), "label": l}

            ds = ds.map(mask_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds.prefetch(prefetch_size)

    if config.data.dataset in ["BRAIN", "FAMLI"]:
        from monai.data import DataLoader

        train_ds = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            shuffle=evaluation == False,
            num_workers=config.data.workers,
            prefetch_factor=2,
            # persistent_workers=True,
            pin_memory=True,
        )
        eval_ds = DataLoader(
            eval_ds,
            batch_size=config.eval.batch_size,
            shuffle=False,
            num_workers=config.data.workers,
            pin_memory=True,
        )

        dataset_builder = None
        # return train_ds, eval_ds, None

    elif config.data.dataset in ["KNEE"]:
        train_ds = create_dataset(train_ds, train_split_name)
        eval_ds = create_dataset(eval_ds, eval_split_name)
    else:
        train_ds = create_dataset(dataset_builder, train_split_name)
        eval_ds = create_dataset(dataset_builder, eval_split_name, val=True)

    #### Test if loader worked
    if config.data.dry_run:
        import matplotlib.pyplot as plt
        import torchvision.transforms.functional as F
        import torchvision

        plt.rcParams["savefig.bbox"] = "tight"

        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]
            fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                im = axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                # fix.colorbar(im, ax=axs[0, i])

        for x in eval_ds:
            print("Shape:", x["image"].shape)
            print(x["image"].numpy().min(), x["image"].numpy().max())
            xt = torch.from_numpy(x["image"].numpy()[:16, ...])
            sns.heatmap(xt[0, 0])
            plt.savefig(f"imgs/BRAIN.png", dpi=200)
            # xt = torch.permute(xt, (0, 3, 1, 2))
            xt = torchvision.utils.make_grid(xt, nrow=4)
            show(xt)
            name = config.data.ood_ds if ood_eval else config.data.dataset
            mask_tag = "-masked" if config.data.mask_marginals else ""
            plt.savefig(
                f"imgs/{name}-c{config.data.select_channel}{mask_tag}.png", dpi=200
            )
            break
        exit()

    return train_ds, eval_ds, dataset_builder
