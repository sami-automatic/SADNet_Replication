import scipy.io
import numpy as np
from PIL import Image
from torch.utils import data

class SIDDTest(data.Dataset):
    
    def __init__(self, noisy_filename, GT_filename):
        super().__init__()
        self.GT = self.read_mat(GT_filename)
        self.noisy = self.read_mat(noisy_filename)



    def __getitem__(self, index):
        gt = self.GT[index]
        noisy = self.noisy[index]
        return gt, noisy

    def __len__(self):
        return len(self.GT)

    def read_mat(self, file_path):
        mat = scipy.io.loadmat(file_path)
        data = mat['ValidationNoisyBlocksSrgb']
        data = np.array(data)

        b_size, blocks, h, w , c = data.shape

        return np.reshape(data,(b_size* blocks,h, w, c))
