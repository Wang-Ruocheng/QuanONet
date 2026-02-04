import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        # [Legacy] 原始实现：使用 torch.rand (0~1均匀分布，全正数)，不减去 0.5
        self.weights1 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)-0.5))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, layers=1, fc_hidden=32):
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width
        self.layers = layers
        
        # 1. Lifting: 2 -> 8 (有足够的特征空间)
        self.fc0 = nn.Linear(2, self.width) 

        # 2. Fourier Layers (动态生成层数)
        self.convs = nn.ModuleList([SpectralConv1d(self.width, self.width, self.modes1) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(layers)])

        # 3. Projection: 8 -> 32 -> 1
        self.fc1 = nn.Linear(self.width, fc_hidden) 
        self.fc2 = nn.Linear(fc_hidden, 1)
        
        # DeepXDE Compatibility (原文件中也有这行)
        self.regularizer = None

    def forward(self, x):
        # Input: (Batch, Sensors, 2)
        x = self.fc0(x)
        x = x.permute(0, 2, 1) # -> (Batch, Width, Sensors)

        for i in range(self.layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.relu(x)

        x = x.permute(0, 2, 1) # -> (Batch, Sensors, Width)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x