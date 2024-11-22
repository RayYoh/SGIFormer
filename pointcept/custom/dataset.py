import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.cache import shared_dict
from pointcept.datasets.scannet import ScanNetDataset
from pointcept.datasets.scannetpp import ScanNetPPDataset
from pointcept.datasets.builder import DATASETS

from pointcept.utils.logger import get_root_logger
from pointcept.datasets.transform import Compose, TRANSFORMS
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


# Custom Dataset
@DATASETS.register_module()
class ScanNetSpDataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
        "superpoint",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        transform=None,
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)
        
        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        
        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if "superpoint" in data_dict.keys():
            data_dict["superpoint"] = (
                data_dict.pop("superpoint").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["superpoint"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index
        return data_dict
    
    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        data_dict = self.transform(data_dict)
        data_dict = self.test_voxelize(data_dict)
        if self.test_crop:
            data_dict = self.test_crop(data_dict)
        data_dict = self.post_transform(data_dict)
        data_dict = dict(
            data_dict=data_dict, segment=segment, 
            instance=instance, name=self.get_data_name(idx)
        )
        return data_dict


@DATASETS.register_module()
class ScanNet200SpDataset(ScanNetSpDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
        "superpoint",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)


from pointcept.custom.scannetpp_constants import CLASS_LABELS_PP, INST_LABELS_PP
@DATASETS.register_module()
class ScanNetppDataset(Dataset):
    """
    Own implementation of ScanNetppDataset based on the official toolkit.
    """
    class2id = np.array([CLASS_LABELS_PP.index(c) for c in INST_LABELS_PP])
    def __init__(
        self,
        split="train",
        data_root="data/scannetpp",
        transform=None,
        lr_file=None,
        la_file=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(ScanNetppDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
        if lr_file:
            self.data_list = [
                os.path.join(data_root, "train", name + ".npy")
                for name in np.loadtxt(lr_file, dtype=str)
            ]
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            data = torch.load(data_path)
            # data = np.load(data_path, allow_pickle=True).item()
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)
        
        if 'vtx' in self.split:
            coord = data["vtx_coords"]
            color = data["vtx_colors"] * 255
            normal = data["vtx_normals"]
            scene_id = data["scene_id"]
            if "vtx_labels" in data.keys():
                segment = data["vtx_labels"].reshape([-1])
            else:
                segment = np.ones(coord.shape[0]) * -1
            if "vtx_instance_labels" in data.keys():
                instance = data["vtx_instance_labels"].reshape([-1])
            else:
                instance = np.ones(coord.shape[0]) * -1
            if "vtx_superpoints" in data.keys():
                superpoint = data["vtx_superpoints"]
            else:
                superpoint = np.ones(coord.shape[0]) * -1
        else:
            coord = data["sampled_coords"]
            color = data["sampled_colors"] * 255
            normal = data["sampled_normals"]
            scene_id = data["scene_id"]
            if "sampled_labels" in data.keys():
                segment = data["sampled_labels"].reshape([-1])
            else:
                segment = np.ones(coord.shape[0]) * -1
            if "sampled_instance_labels" in data.keys():
                instance = data["sampled_instance_labels"].reshape([-1])
            else:
                instance = np.ones(coord.shape[0]) * -1
            if "sampled_superpoints" in data.keys():
                superpoint = data["sampled_superpoints"]
            else:
                superpoint = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            color=color,
            normal=normal,
            segment=segment,
            instance=instance,
            superpoint=superpoint,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(segment).astype(np.bool)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        data_dict = self.transform(data_dict)
        data_dict = self.test_voxelize(data_dict)
        if self.test_crop:
            data_dict = self.test_crop(data_dict)
        data_dict = self.post_transform(data_dict)
        data_dict = dict(
            data_dict=data_dict, segment=segment, 
            instance=instance, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ScanNetPPSpDataset(ScanNetPPDataset):
    """
    ScanNetPPDataset is provided by Pointcept.
    """
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "instance",
        "superpoint",
    ]
    class2id = np.array([CLASS_LABELS_PP.index(c) for c in INST_LABELS_PP])
    
    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "superpoint" in data_dict.keys():
            data_dict["superpoint"] = data_dict["superpoint"].astype(np.int32)

        if not self.multilabel:
            if "segment" in data_dict.keys():
                if "vtx" in self.split:
                    data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
                else:
                    data_dict["segment"] = data_dict["segment"][:, 0].astype(np.int32)
            else:
                data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )

            if "instance" in data_dict.keys():
                if "vtx" in self.split:
                    data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
                else:
                    data_dict["instance"] = data_dict["instance"][:, 0].astype(np.int32)
            else:
                data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )
        else:
            raise NotImplementedError
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        data_dict = self.transform(data_dict)
        data_dict = self.test_voxelize(data_dict)
        if self.test_crop:
            data_dict = self.test_crop(data_dict)
        data_dict = self.post_transform(data_dict)
        data_dict = dict(
            data_dict=data_dict, segment=segment, 
            instance=instance, name=self.get_data_name(idx)
        )
        return data_dict

