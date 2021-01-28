from torch.utils.data import DataLoader
from torch.autograd import Variable

import wandb
from datasets.dataloader import *
from utils.train_utils import *
from utils.config import *
from models.sadNet import SADNET


def train(cfg):
    wandb.init(project='sad_net_replicate', config=cfg)

    if cfg.real:
        dataset = Dataset_h5_real(cfg.src, patch_size=cfg.patch_size, train=True)
        dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    else:
        dataset = Dataset_from_h5(cfg.src, sigma=cfg.sigma, gray=False,
                                  transform=transforms.Compose(
                                      [transforms.RandomCrop((128, 128)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.Lambda(lambda img: RandomRot(img)),
                                       transforms.ToTensor()
                                       # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
                                       ]),
                                  )
        dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=cfg.shuffle, num_workers=4,
                                drop_last=True)

    # Build model
    model = SADNET(cfg.num_channel, cfg.offset_channel)

    # Loss
    criterion = torch.nn.MSELoss()

    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            model = torch.nn.DataParallel(model).cuda()
            criterion = criterion.cuda()
        else:
            model.to(device)  # = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(0, cfg.num_epoch):

        loss_sum = 0
        step_lr_adjust(optimizer, epoch, init_lr=cfg.lr, step_size=cfg.step_size, gamma=cfg.GAMMA)
        print('Epoch {}, lr {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        start_time = time.time()

        for i, data in enumerate(dataloader):
            num_step = epoch * len(dataloader) + i

            input, label = data
            if torch.cuda.is_available():
                input, label = input.to(device), label.to(device)  # input.cuda(), label.cuda()
            input, label = Variable(input), Variable(label)
            # print(input.size(), label.size())
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if (i % cfg.wandb_interval == 0) and (i != 0):
                wandb.log({"epoch": epoch, "loss": loss}, step=num_step)
                wandb.log({"examples": [wandb.Image(transforms.ToPILImage()(input.cpu()[0]),
                                                    caption="noise"),
                                        wandb.Image(transforms.ToPILImage()(torch.clamp(output, min=0., max=1.).cpu()[0]),
                                                    caption="output"),
                                        wandb.Image(transforms.ToPILImage()(label.cpu()[0]),
                                                    caption="GT"), ]}
                          )
                loss_avg = loss_sum / 100
                loss_sum = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f} Time: {:4.4f}s".format(
                    epoch + 1, cfg.num_epoch, i + 1, len(dataloader), loss_avg, time.time() - start_time))
                start_time = time.time()

        # save model
        if epoch % cfg.save_interval == 0:
            model_name = "model_dict_sigma{}_epoch{}.pth".format(cfg.sigma, epoch)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(cfg.ckpt_dir, model_name))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, model_name))


if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cfg = argument_parser().parse_args()
    device = torch.device('cuda')
    create_dir(cfg.ckpt_dir)
    train(cfg)
