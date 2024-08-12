import torch

from PIL import Image
from pathlib import Path
from einops import rearrange


class CelebA:
    mode2dirname = {
        "rgb": "rgb",
        "sketch": "sketch",
        "normal": "normal",
    }

    def __init__(self, root, train, transform_dict, modes, ratio, size=32, repeat_factor=1):
        assert size in [32, 64, 128, 256, 512]
        if train:
            self.data_dir = Path(root) / f"train/{size}"
        else:
            self.data_dir = Path(root) / f"test/{size}"
        self.transform_dict = transform_dict
        self.modes = modes
        self.ratio = ratio

        self.data_path = {}
        for mode in self.modes:
            all_paths = (self.data_dir / self.mode2dirname[mode]).iterdir()
            self.data_path[mode] = sorted(all_paths)
            self.data_path[mode] = self.data_path[mode][: int(len(self.data_path[mode]) * ratio)]

        self.len_ori = self.len = len(self.data_path[mode])

        if not train:
            import warnings

            warnings.warn(f"meta-test set is replicated {repeat_factor} times")
            self.len *= repeat_factor
            self.len = int(self.len)

        self[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = idx % self.len_ori

        data_dict = {}
        validity = []
        for mode in self.data_path:
            path = self.data_path[mode][idx]
            data = Image.open(path)
            data_dict[mode] = self.transform_dict[mode](data)

            validity += ["_".join(path.stem.split("_")[:4])]

        data_dict["idx"] = idx

        assert len(set(validity)) == 1

        return data_dict, 0


class CelebAMeta:
    mode2dirname = {
        "rgb": "rgb",
        "sketch": "sketch",
        "normal": "normal",
    }

    def __init__(self, root, modes, ratio, size=32, repeat_factor=1):
        assert size in [32, 64, 128, 256, 512]
        self.modes = modes
        self.size = size

        self.data_dir = Path(root) / f"test/{size}"

        self.data_path = {}
        for mode in self.modes[:1]:
            all_paths = (self.data_dir / self.mode2dirname[mode]).iterdir()
            self.data_path[mode] = sorted(all_paths)
            self.data_path[mode] = self.data_path[mode][: int(len(self.data_path[mode]) * ratio)]
        self.len_ori = len(self.data_path[mode])

        self.data_path = Path(root) / "meta-test" / f"{size}" / f"nr:{repeat_factor}.pt"
        self.data = torch.load(self.data_path, map_location="cpu")
        self.len = len(self.data["xs_support_dict"][self.modes[0]])
        assert self.len % self.len_ori == 0

        self[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # idx_ori = idx % self.len_ori

        data_dict = {
            "xs_support_dict": {},
            "ys_support_dict": {},
            "ms_support_dict": {},
            "xs_query_dict": {},
            "ys_query_dict": {},
            "ids_sample_dict": {},
            "ids_restore_dict": {},
        }

        for mode in self.modes:
            data_dict["xs_support_dict"][mode] = self.data["xs_support_dict"][mode][idx]
            data_dict["ys_support_dict"][mode] = self.data["ys_support_dict"][mode][idx]
            data_dict["ms_support_dict"][mode] = self.data["ms_support_dict"][mode][idx]
            data_dict["xs_query_dict"][mode] = self.data["xs_query_dict"][mode][idx]
            data_dict["ys_query_dict"][mode] = self.data["ys_query_dict"][mode][idx]
            data_dict["ids_restore_dict"][mode] = self.data["ids_restore_dict"][mode][idx]
            data_dict["ids_sample_dict"][mode] = self.data["ids_sample_dict"][mode][idx]

            data_dict[mode] = rearrange(data_dict["ys_query_dict"][mode], "(h w) d -> d h w", h=self.size, w=self.size)

        return data_dict, 0
