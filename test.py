import torch
import numpy as np
import glog as log
from torchvision import transforms

from datasets import SIDDTest
from sadNet import SADNET
from SSIM import SSIM

model_path = "ckpt/SADNET/final_model_sadnet.pth"
dataset = SIDDTest("ValidationGtBlocksSrgb.mat", "ValidationNoisyBlocksSrgb.mat")
image_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


PSNR = PSNR()
SSIM = SSIM()  # kornia's SSIM is buggy.

model = SADNET()
model_dict = torch.load(model_path)
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()


def eval():
    psnr_lst, ssim_lst = list(), list()
    with torch.no_grad():
        for batch_idx, (gt, noisy) in enumerate(image_loader):
            gt = gt.float().cuda()
            noisy = noisy.float().cuda()

            to_pil = transforms.ToPILImage()
            cropper = transforms.TenCrop(128)
            lmbda = transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))

            gt = lmbda(cropper(to_pil(gt.squeeze(0).cpu()))).cuda()
            noisy = lmbda(cropper(to_pil(noisy.squeeze(0).cpu()))).cuda()

            prediction = model(noisy)
            prediction = torch.clamp(prediction, max=1., min=0.)
            if batch_idx == 0:
                transforms.ToPILImage()(prediction[1].cpu()).show(title="out")
                transforms.ToPILImage()(gt[1].cpu()).show(title="gt")
                transforms.ToPILImage()(noisy[1].cpu()).show(title="noisy")

            ssim = SSIM(gt, prediction).item()
            ssim_lst.append(ssim)

            psnr = PSNR(255.*gt, 255.*prediction).item()
            psnr_lst.append(psnr)

            # log.info(" \tSSIM: {}\tPSNR: {}".format(batch_idx, len(image_loader), round(ssim, 3), round(psnr, 3)))
    results = {"Dataset": "SIDD", "PSNR": np.mean(psnr_lst), "SSIM": np.mean(ssim_lst)}
    print(results)


if __name__ == '__main__':
    eval()
