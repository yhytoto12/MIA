import torch
from einops import rearrange
from pathlib import Path


class Synthetic(torch.utils.data.Dataset):
    def __init__(self, root, train, modes, ratio, use_jitter=False, repeat_factor=1):
        super().__init__()
        self.data_dir = Path(root)
        self.train = train
        self.modes = modes
        self.ratio = ratio
        self.use_jitter = use_jitter

        if train:
            self.data = torch.load(self.data_dir / "meta-dataset.pt")
        else:
            self.data = torch.load(self.data_dir / "meta-dataset.pt")
            self.eval_keys = list(self.data["ms_mtest_dict_main"].keys())

        self.len = self.len_ori = self.data["xs_mtrain"].shape[0] if self.train else self.data["xs_mtest"].shape[0]

        if not train:
            import warnings

            warnings.warn(f"meta-test set is replicated {repeat_factor} times")
            self.len *= repeat_factor
            self.len = int(self.len)

        self[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len_ori
        data_dict, info_dict = {}, {}

        if self.train:
            for mode in self.data["ys_mtrain_dict_main"].keys():
                # shape of [channel, num_points]
                data_dict[mode] = {}
                info_dict[mode] = {}
                if self.use_jitter:
                    data_dict[mode] = self.data["ys_mtrain_jittered_dict_main"][mode][index].unsqueeze(0)
                    info_dict[mode]["a"] = self.data["ps_mtrain_jittered_main"][mode][index, 0]
                else:
                    data_dict[mode] = self.data["ys_mtrain_dict_main"][mode][index].unsqueeze(0)
                    info_dict[mode]["a"] = self.data["ps_mtrain_main"][mode][index, 0]

        else:
            info_dict = {
                "ids_keep": {k: self.data["ids_keep_main"][k][index] for k in self.data["ids_keep_main"].keys()}
            }
            for mode in self.data["ys_mtest_dict_main"].keys():
                info_dict[mode] = {
                    "mask": {
                        k: self.data["ms_mtest_dict_main"][k][mode][index]
                        for k in self.data["ms_mtest_dict_main"].keys()
                    }
                }
                if self.use_jitter:
                    data_dict[mode] = self.data["ys_mtest_jittered_dict_main"][mode][index].unsqueeze(0)
                    info_dict[mode]["a"] = self.data["ps_mtest_jittered_main"][mode][index, 0]
                else:
                    data_dict[mode] = self.data["ys_mtest_dict_main"][mode][index].unsqueeze(0)
                    info_dict[mode]["a"] = self.data["ps_mtest_main"][mode][index, 0]

        return data_dict, info_dict


class SyntheticMeta(torch.utils.data.Dataset):
    def __init__(self, root, modes, ratio, use_jitter=False, repeat_factor=1):
        super().__init__()
        assert use_jitter is True
        self.modes = modes

        self.data_dir = Path(root)
        self.data_ori = torch.load(self.data_dir / "meta-dataset.pt")
        self.len_ori = len(self.data_ori["xs_mtest"])

        self.data_path = Path(root) / "meta-test" / f"nr:{repeat_factor}.pt"
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
            "info_dict": {mode: {} for mode in self.modes},
        }

        for mode in self.modes:
            data_dict["xs_support_dict"][mode] = self.data["xs_support_dict"][mode][idx]
            data_dict["ys_support_dict"][mode] = self.data["ys_support_dict"][mode][idx]
            data_dict["ms_support_dict"][mode] = self.data["ms_support_dict"][mode][idx]
            data_dict["xs_query_dict"][mode] = self.data["xs_query_dict"][mode][idx]
            data_dict["ys_query_dict"][mode] = self.data["ys_query_dict"][mode][idx]
            data_dict["ids_restore_dict"][mode] = self.data["ids_restore_dict"][mode][idx]
            data_dict["ids_sample_dict"][mode] = self.data["ids_sample_dict"][mode][idx]
            data_dict["info_dict"][mode]["a"] = self.data["info_dict"][mode]["a"][idx]

            data_dict[mode] = rearrange(data_dict["ys_query_dict"][mode], "n d -> d n")

        return data_dict, 0
