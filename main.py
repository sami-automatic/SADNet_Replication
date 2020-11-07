from models.sadNet import SADNET
import torch
from torch.autograd import Variable

if __name__ == '__main__':
    device = torch.device('cuda')
    print(device)
    x = torch.rand((2,3,128,128))
    print(x.shape)
    print(x)

    model = SADNET(32, 32)
    model.to(device)
    x = x.to(device)

    x = Variable(x)
    model.train()
    out = model(x)
    print(out.shape)
    print(out)
