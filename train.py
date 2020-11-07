import os
import numpy as np
import time
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

import matplotlib.image as mpimg
from skimage.measure import compare_psnr, compare_ssim

import wandb
from dataloader import *
from utils import *
from models.sadNet import SADNET


src_path = "./dataset/train/SIDD_RENOIR_h5/train.h5"
ckpt_dir = "./ckpt/SADNET/"

save_epoch = 1 #save model per every N epochs
patch_size = 128
batch_size = 2
val_patch_size = 512

lr = 1e-4
N_EPOCH = 200 #number of training epochs
MILESTONE = 60 #the epochs for weight decay
GAMMA = 0.5
n_channel, offset_channel = 32, 32

cfg = dict(
    epochs=N_EPOCH,
    batch_size=batch_size,
    learning_rate=lr,
    dataset="RENOIR and SIDD"
    )

def train():
    wandb.init(project='sad_net_replicate', config=cfg)
    
    # Load dataset
    dataset = Dataset_h5_real(src_path, patch_size=patch_size, train=True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # Build model
    model = SADNET(n_channel, offset_channel)

    #Loss
    criterion = torch.nn.MSELoss()
   
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            #model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            model = torch.nn.DataParallel(model).cuda()
            criterion = criterion.cuda()
        else:
            model.to(device) # = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(0, N_EPOCH):

        loss_sum = 0
        step_lr_adjust(optimizer, epoch, init_lr=lr, step_size=MILESTONE, gamma=GAMMA)
        print('Epoch {}, lr {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        start_time = time.time()
        
        for i, data in enumerate(dataloader):
            num_step = epoch * len(dataloader) + i

            input, label = data
            if torch.cuda.is_available():
                input, label = input.to(device), label.to(device) #input.cuda(), label.cuda()
            input, label = Variable(input), Variable(label)

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if (i % 1 == 0) and (i != 0) :
                wandb.log({"epoch": epoch, "loss": loss}, step=num_step)
                wandb.log({"examples": [wandb.Image(transforms.ToPILImage()(input.cpu()[0]),
                                                                  caption="noise"),
                                        wandb.Image(transforms.ToPILImage()(output.cpu()[0]),
                                                                 caption="output"),
                                        wandb.Image(transforms.ToPILImage()(label.cpu()[0]),
                                                                 caption="GT"),]}
                            )
                loss_avg = loss_sum / 100
                loss_sum = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f} Time: {:4.4f}s".format(
                    epoch + 1, N_EPOCH, i + 1, len(dataloader), loss_avg, time.time()-start_time))
                start_time = time.time()

        # save model
        if epoch % save_epoch == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))


if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device('cuda')
    create_dir(ckpt_dir)
    train()
