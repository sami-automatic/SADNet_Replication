import os
import numpy as np
import time
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import matplotlib.image as mpimg
from skimage.measure import compare_psnr, compare_ssim

from dataloader import *
from utils import *
from models.sadNet import SADNET

def train():
    log_dir = "./log/SADNET_color_sig50/"
    src_path = "./dataset/train/SIDD_RENOIR_h5/train.h5"
    val_path = "./dataset/test/color_sig50/valid.h5"
    ckpt_dir = "./ckpt/SADNET_color_sig50/"
    log_dir = "./log/SADNET_color_sig50/"
    save_val_img = True #save the last validated image for comparison
    save_epoch = 1 #save model per every N epochs
    patch_size = 128
    batch_size = 16
    val_patch_size = 512

    lr = 1e-4
    N_EPOCH = 200 #number of training epochs
    MILESTONE = 60 #the epochs for weight decay
    GAMMA = 0.5
    n_channel, offset_channel = 32, 32

    # Load dataset
    dataset = Dataset_h5_real(src_path, patch_size=patch_size, train=True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    
    dataset_val = Dataset_h5_real(src_path=val_path, patch_size=val_patch_size, gray=False, train=False)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    
    # Build model
    model = SADNET(n_channel, offset_channel)

    #Loss
    criterion = torch.nn.MSELoss()
   

    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            #model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            model = torch.nn.DataParallel(model).cuda()
            criterion = criterion.cuda()
        else:
            model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=milestone, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
    writer = SummaryWriter(log_dir)

    for epoch in range(0, N_EPOCH):

        loss_sum = 0
        step_lr_adjust(optimizer, epoch, init_lr=lr, step_size=MILESTONE, gamma=GAMMA)
        print('Epoch {}, lr {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        start_time = time.time()
        for i, data in enumerate(dataloader):
            input, label = data
            if torch.cuda.is_available():
                input, label = input.cuda(), label.cuda()
            input, label = Variable(input), Variable(label)

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if (i % 100 == 0) and (i != 0) :
                loss_avg = loss_sum / 100
                loss_sum = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f} Time: {:4.4f}s".format(
                    epoch + 1, N_EPOCH, i + 1, len(dataloader), loss_avg, time.time()-start_time))
                start_time = time.time()
                # Record train loss
                writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
                # Record learning rate
                #writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
                writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        # save model
        if epoch % args.save_epoch == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))

        # validation
        if val_path:
            if epoch % args.val_epoch == 0:
                psnr = 0
                loss_val = 0
                model.eval()
                for i, data in enumerate(dataloader_val):
                    input, label = data
                    if torch.cuda.is_available():
                        input, label = input.cuda(), label.cuda()
                    input, label = Variable(input), Variable(label)

                    test_out = model(input)
                    test_out.detach_()

                    # 计算loss
                    loss_val += criterion(test_out, label).item()
                    rgb_out = test_out.cpu().numpy().transpose((0,2,3,1))
                    clean = label.cpu().numpy().transpose((0,2,3,1))
                    for num in range(rgb_out.shape[0]):
                        denoised = np.clip(rgb_out[num], 0, 1)
                        psnr += compare_psnr(clean[num], denoised)
                img_nums = rgb_out.shape[0] * len(dataloader_val)
                #img_nums = batch_size * len(dataloader_val)
                psnr = psnr / img_nums
                loss_val = loss_val / len(dataloader_val)
                print('Validating: {:0>3} , loss: {:.8f}, PSNR: {:4.4f}'.format(img_nums, loss_val, psnr))
                #mpimg.imsave(ckpt_dir+"img/%04d_denoised.png" % epoch, rgb_out[0])
                writer.add_scalars('Loss_group', {'valid_loss': loss_val}, epoch)
                writer.add_scalar('valid_psnr', psnr, epoch)
                if save_val_img:
                    mpimg.imsave(ckpt_dir+"img/%04d_denoised.png" % epoch, denoised)
                        


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = 1

    create_dir(log_dir)
    create_dir(ckpt_dir)
    if save_val_img:
        create_dir(ckpt_dir+'img/')
    train()
