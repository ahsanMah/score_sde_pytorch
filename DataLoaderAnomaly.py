import os
import time

import pandas as pd
import nrrd
import numpy as np

import torch
from torch.utils.data import Dataset

from monai.data import CacheDataset

from torch.utils.data import DataLoader

# from monai.data import DataLoader

from monai.transforms import (
    ToTensor,
    LoadImage,
    Lambda,
    AddChannel,
    RepeatChannel,
    ScaleIntensityRange,
    ScaleIntensityRangePercentiles,
    SpatialCrop,
    RandSpatialCrop,
    RandFlip,
    RandRotate,
    Compose,
    BorderPad,
    RandZoom,
    RandGaussianSmooth,
    RandGaussianNoise,
    RandAdjustContrast,
    TorchVision,
    RandLambda,
)
from mri_utils import RandTumor


class AnomalyDataLoader(Dataset):
    def __init__(
        self, df, mount_point="./", transform=None, device="cpu", random_last=5
    ):
        self.df = df
        self.mount = mount_point
        self.transform = transform
        self.device = device
        self.random_last = random_last

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):

        row = self.df.loc[idx]
        img_path = row.file_path
        ga = row.ga_boe
        tag = row.tag
        # ga_bin = row.ga_bin
        try:
            img, header = nrrd.read(os.path.join(self.mount, img_path), index_order="C")
            img = torch.tensor(img, dtype=torch.float, device=self.device)
            assert len(img.shape) == 3
            assert img.shape[1] == 256
            assert img.shape[2] == 256
        except:
            print("Error reading cine: " + img_path)
            img = torch.zeros(200, 256, 256)

        img = img[img.shape[0] - np.random.randint(1, self.random_last)]

        if self.transform:
            img = self.transform(img)

        # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize = T.Normalize(mean=[0.485], std=[0.229])
        # print(img.shape)
        # img = normalize(img)

        return dict(image=img, label=np.array([ga]))


class TransformFrames:
    def __init__(self, training=True, ood=False):

        self.debug_transform = Compose(
            [
                AddChannel(),
                ScaleIntensityRangePercentiles(1, 99, 0, 1.0, clip=True),
                ToTensor(),
            ]
        )

        if training:
            self.transform = Compose(
                [
                    #        Lambda(func=random_choice),
                    AddChannel(),
                    ScaleIntensityRangePercentiles(1, 99, 0, 1.0, clip=True),
                    BorderPad(spatial_border=[32, 32]),
                    RandSpatialCrop(roi_size=[256, 256], random_size=False),
                    RandFlip(prob=0.5, spatial_axis=1),
                    RandAdjustContrast(prob=0.2, gamma=(0.5, 1.5)),
                    # RandGaussianSmooth(
                    #     sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5), prob=0.5, approx="erf"
                    # ),
                    RandRotate(prob=0.2, range_x=3.14),
                    RandZoom(prob=0.2, min_zoom=0.8, max_zoom=1.2),
                    # RepeatChannel(repeats=3),
                    ToTensor(),
                    # TorchVision(
                    #     "image",
                    #     "Normalize",
                    #     mean=[0.485, 0.456, 0.406],
                    #     std=[0.229, 0.224, 0.225],
                    # ),
                    # normalize
                ]
            )
        elif ood:
            deformer = RandTumor(
                spacing=1.0,
                max_tumor_size=50.0,
                magnitude_range=(35.0, 50.0),
                prob=1.0,
                spatial_size=(256, 256),  # [168, 200, 152],
                padding_mode="zeros",
            )

            deformer.set_random_state(seed=42)

            self.transform = Compose(
                [
                    AddChannel(),
                    # ScaleIntensityRange(0.0, 255.0, 0, 1.0),
                    ScaleIntensityRangePercentiles(1, 99, 0, 1.0, clip=True),
                    RandLambda(deformer),
                    ToTensor(),
                ]
            )
        else:
            self.transform = Compose(
                [
                    #    Lambda(func=random_choice),
                    AddChannel(),
                    # BorderPad(spatial_border=[-1, 24, 24]),
                    # RandSpatialCrop(roi_size=[-1, 256, 256], random_size=False),
                    # Flip(spatial_axis=3),
                    # ScaleIntensityRange(0.0, 255.0, 0, 1.0),
                    ScaleIntensityRangePercentiles(1, 99, 0, 1.0, clip=True),
                    # RepeatChannel(repeats=3),
                    ToTensor(),
                    # TorchVision(
                    #     "Normalize",
                    #     mean=[0.485, 0.456, 0.406],
                    #     std=[0.229, 0.224, 0.225],
                    # ),
                    # Lambda(lambda x: torch.transpose(x, 0, 1)),
                    # ToTensor(),
                    # normalize
                ]
            )

    def __call__(self, x):
        x = self.transform(x)
        # x = self.transform(torch.rand_like(x))
        # x = torch.abs(self.debug_transform(x) - self.transform(x))
        return x


if __name__ == "__main__":

    # dataset_dir = "/work/amahmood/data/GWH/Groups/FAMLI/Shared/C1_ML_Analysis/CSV_files"
    dataset_dir = "/FAMLI/Shared/C1_ML_Analysis/"
    train_df = pd.read_csv(
        os.path.join(
            dataset_dir,
            "CSV_files",
            "ALL_C1_C2_cines_gt_ga_withmeta_masked_resampled_256_spc075_uuid_study_flyto_uuid_train.csv",
        )
    )
    print(np.unique(train_df.tag, return_counts=True))

    train_df = train_df.loc[train_df["tag"].isin(["BPD", "HC"])]
    train_df = train_df.loc[:, ["file_path", "ga_boe", "tag"]]
    train_df = train_df.query("126 <= ga_boe and ga_boe <= 210").reset_index(drop=True)

    orig_val_df = pd.read_csv(
        os.path.join(
            dataset_dir,
            "CSV_files",
            "ALL_C1_C2_cines_gt_ga_withmeta_masked_resampled_256_spc075_uuid_study_flyto_uuid_valid.csv",
        )
    )
    print(np.unique(orig_val_df.tag, return_counts=True))
    print(orig_val_df.tag.value_counts())

    INLIER_TAGS = ["BPD", "HC"]
    OOD_TAGS = ["FL"]

    val_df = orig_val_df.loc[orig_val_df["tag"].isin(INLIER_TAGS)]
    val_df = val_df.loc[:, ["file_path", "ga_boe", "tag"]]
    val_df = val_df.query("126 <= ga_boe and ga_boe <= 210").reset_index(drop=True)

    ood_df = orig_val_df.loc[orig_val_df["tag"].isin(OOD_TAGS)]
    ood_df = ood_df.loc[:, ["file_path", "ga_boe", "tag"]]
    ood_df = ood_df.query("126 <= ga_boe and ga_boe <= 210").reset_index(drop=True)

    training_data = AnomalyDataLoader(
        train_df, mount_point=dataset_dir, transform=TransformFrames(training=True)
    )

    train_dataloader = DataLoader(
        training_data,
        batch_size=2,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=False,
    )

    val_data = AnomalyDataLoader(
        val_df, mount_point=dataset_dir, transform=TransformFrames(training=False)
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=32,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=False,
    )

    # nsamples = 0
    # print("Starting train set iter")
    # for i, batch in enumerate(train_dataloader):
    #     nsamples += batch["image"].shape[0]
    #     break
    # print(f"Collected {nsamples} from train set")

    nsamples = 0
    print("Starting val set iter")
    for i, batch in enumerate(val_dataloader):
        nsamples += batch["image"].shape[0]
    print(f"Collected {nsamples} from val set")

    nsamples = 0
    ood_data = AnomalyDataLoader(
        ood_df, mount_point=dataset_dir, transform=TransformFrames(training=False)
    )
    ood_dataloader = DataLoader(
        ood_data,
        batch_size=32,
        num_workers=6,
        prefetch_factor=4,
        pin_memory=False,
    )
    print("Starting ood set iter")
    for i, batch in enumerate(ood_dataloader):
        nsamples += batch["image"].shape[0]
    print(f"Collected {nsamples} from ood set")


#     train_df = train_df.loc[train_df["tag"].isin(["BPD", "HC"])]
