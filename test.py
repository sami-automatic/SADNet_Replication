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


def eval_DND( data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch 
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:pass

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3],:].copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            
            for yy in range(2):
                for xx in range(2):
                    Idenoised_crop = model(Inoisy_crop)
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k+1, 20))
        print('[%d/%d] %s done\n' % (i+1, 50, filename))

if __name__ == '__main__':
    #eval()
