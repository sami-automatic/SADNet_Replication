import torch

from utils.train_utils import *
from utils.config import *
from engine.train import train

if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cfg = argument_parser().parse_args()
    device = torch.device('cuda')
    create_dir(cfg.ckpt_dir)
    train(cfg)

