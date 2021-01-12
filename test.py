import torch
import kornia

from datasets import SIDDTest
from sadNet import SADNET
from SSIM import SSIM

model_path = ""

dataset = SIDDTest("ValidationGtBlocksSrgb.mat", "ValidationNoisyBlocksSrgb.mat")
image_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=False)

PSNR = kornia.losses.psnr.PSNRLoss(max_val=1.)
SSIM = SSIM()  # kornia's SSIM is buggy.

model = SADNET()
model_dict = torch.load(model_path)
model.load_state_dict(model_dict)

model = model.cuda()
def eval():
    psnr_lst, ssim_lst = list(), list()
    with torch.no_grad():
        for batch_idx, (gt, noisy) in enumerate(image_loader):
            gt = gt.float().cuda()
            noisy = noisy.float().cuda()

            batch_size, channels, h, w = gt.size()

            prediction = model(noisy)

            prediction = torch.clamp(prediction, max=1., min=0.)

            ssim = SSIM(255. * gt, 255. * prediciton).item()
            ssim_lst.append(ssim)

            psnr = PSNR(gt, prediction).item()
            psnr_lst.append(psnr)

            log.info(" \tSSIM: {}\tPSNR: {}".format(batch_idx, len(image_loader), round(ssim, 3), round(psnr, 3)))

            