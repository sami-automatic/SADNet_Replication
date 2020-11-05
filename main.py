from models.sadNet import SADNET
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    print(device)
    x = torch.rand((2,3,256,256))
    print(x.shape)
    print(x)

    model = SADNET(32, 32)
    model.to(device)
    x = x.to(device)
    out = model(x)
    print(out.shape)
    print(out)
