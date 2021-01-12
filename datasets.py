import scipy.io
import numpy as np
import torch
from torch.utils import data


class SIDDTest(data.Dataset):
    def __init__(self, GT_filename, noisy_filename):
        super().__init__()
        self.GT = self.read_mat(GT_filename)
        self.noisy = self.read_mat(noisy_filename)

    def __getitem__(self, index):
        gt = self.GT[index]
        gt = torch.from_numpy(np.ascontiguousarray(np.transpose(gt, (2, 0, 1)))).float() / 255.

        noisy = self.noisy[index]
        noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(noisy, (2, 0, 1)))).float() / 255.
        return gt, noisy

    def __len__(self):
        return len(self.GT)

    def read_mat(self, file_path):
        mat = scipy.io.loadmat(file_path)

        key = file_path.split(".")[0]
        data = mat[key]
        data = np.array(data)

        b_size, blocks, h, w, c = data.shape

        return np.reshape(data, (b_size * blocks, h, w, c))
