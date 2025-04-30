import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1,
                 start_with_relu=True, grow_first=True, with_bn=True):
        super(Block, self).__init__()

        self.with_bn = with_bn

        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            if with_bn:
                self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < (reps - 1) else out_channels
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            if with_bn:
                rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU()

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            if self.with_bn:
                skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class build_fakesensor(nn.Module):
    def __init__(self, inp_cs=[256, 256, 256, 64, 16], oup_c=256, mask = False):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.se0 = nn.Sequential(nn.Linear(inp_cs[0], inp_cs[0]//16),
                                 nn.ReLU(),
                                 nn.Linear(inp_cs[0]//16, oup_c),
                                 nn.Sigmoid())
        
        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.decoder2 = Block(256, 256, 1, 1)
        self.se1 = nn.Sequential(nn.Linear(inp_cs[1], inp_cs[1]//16),
                                 nn.ReLU(),
                                 nn.Linear(inp_cs[1]//16, oup_c),
                                 nn.Sigmoid())
        
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder4 = Block(256, 256, 1, 1)
        self.se2 = nn.Sequential(nn.Linear(inp_cs[2], inp_cs[2]//16),
                                 nn.ReLU(),
                                 nn.Linear(inp_cs[2]//16, oup_c),
                                 nn.Sigmoid())
        

        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder6 = Block(256, 256, 1, 1)
     
        # self.decoder7 = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     SeparableConv2d(128, 64, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.decoder8 = Block(64, 64, 1, 1)

        # self.decoder9 = nn.Sequential(
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     SeparableConv2d(64, 16, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU()
        # )
        # self.decoder10 = Block(16, 16, 1, 1)

        if not mask:
            self.decoder7 = nn.Sequential(
                nn.Conv2d(256, 3, 1, 1, bias=False),
            )
        else:
            self.decoder7 = nn.Sequential(
                nn.Conv2d(256, 1, 1, 1, bias=False),
                nn.Sigmoid()
            )
        print('building fake sensor...')
        
    def forward(self, x):
        B = x.size(0)
        att0 = self.se0(self.avg(x).flatten(1))[:, :, None, None]
        x = self.decoder2(self.decoder1(x))
        att1 = self.se1(self.avg(x).flatten(1))[:, :, None, None]
        x = self.decoder4(self.decoder3(x))
        att2 = self.se2(self.avg(x).flatten(1))[:, :, None, None]
        x = self.decoder6(self.decoder5(x))
        # x = self.decoder8(self.decoder7(x))
        # x = self.decoder9(x)
        x = self.decoder7(x)

        # return x, [att0, att1, att2]
        return x, [att0, att1, att2]


class DropBlock(nn.Module):
    def __init__(self, block_size=3, keep_prob=0.9):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)
    
    def reset(self, block_size, keep_prob):
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.gamma = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size//2, block_size//2)

    def calculate_gamma(self, x):
        return  (1-self.keep_prob) * x.shape[-1]**2/\
                (self.block_size**2 * (x.shape[-1] - self.block_size + 1)**2) 

    def forward(self, x):
        if (not self.training or self.keep_prob==1): #set keep_prob=1 to turn off dropblock
            return x
        if self.gamma is None:
            self.gamma = self.calculate_gamma(x)
        if x.type() == 'torch.cuda.HalfTensor': #TODO: not fully support for FP16 now 
            FP16 = True
            x = x.float()
        else:
            FP16 = False
        p = torch.ones_like(x) * (self.gamma)
        mask = 1 - torch.nn.functional.max_pool2d(torch.bernoulli(p),
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.padding)

        out =  mask * x * (mask.numel()/mask.sum())

        if FP16:
            out = out.half()
        return out
    
if __name__ == "__main__":
    x = torch.randn(2, 256, 8, 8)
    fs = build_fakesensor()
    y, attns = fs(x)
    for i in attns:
        print(i.shape)
    print(y.shape)