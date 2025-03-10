import torch
from torch.utils.data import Dataset


class SplitDataset(Dataset):
    def __init__(self, data, configs):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.configs = configs

    def __getitem__(self, item):
        # (b, 64, 64, 1)
        inputs = self.data[item: item + self.configs.total_length]
        mask_true = torch.zeros((self.configs.pred_length - 1, self.configs.img_width, self.configs.img_width, 1),
                                dtype=torch.float32)
        return inputs, mask_true

    def __len__(self):
        return len(self.data) - self.configs.total_length
