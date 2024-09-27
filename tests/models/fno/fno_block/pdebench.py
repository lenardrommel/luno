"""Reference implementation of 1D, 2D, and 3D FNO blocks from PDEBench."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import pytest

from numpy.typing import ArrayLike


def spectal_convolution_from_nola_weights(
    R: ArrayLike,
) -> SpectralConv1d | SpectralConv2d_fast | SpectralConv3d:
    R = np.asarray(R)

    assert 3 <= R.ndim <= 5
    assert all(n % 2 == 0 for n in R.shape[:-3])

    modes = tuple(n // 2 for n in R.shape[:-3]) + (R.shape[-3],)
    out_channels, in_channels = R.shape[-2:]

    if len(modes) == 1:
        SpectralConv = SpectralConv1d
    elif len(modes) == 2:
        SpectralConv = SpectralConv2d_fast
    elif len(modes) == 3:
        SpectralConv = SpectralConv3d
    else:
        raise ValueError("Only 1D, 2D and 3D supported")

    conv = SpectralConv(in_channels, out_channels, *modes)

    R = np.moveaxis(R, -1, 0)  # shape: (in_channels, *modes, out_channels)
    R = np.moveaxis(R, -1, 1)  # shape: (in_channels, out_channels, *modes)
    R = torch.tensor(R)

    if len(modes) == 1:
        conv.weights1.data = R
    elif len(modes) == 2:
        conv.weights1.data = R[:, :, : modes[0], :]
        conv.weights1.data = R[:, :, -modes[0] :, :]
    else:
        assert len(modes) == 3

        conv.weights1.data = R[:, :, : modes[0], : modes[1], :]
        conv.weights2.data = R[:, :, -modes[0] :, : modes[1], :]
        conv.weights3.data = R[:, :, : modes[0], -modes[1] :, :]
        conv.weights4.data = R[:, :, -modes[0] :, -modes[1] :, :]

    return conv


class FNOBlock(nn.Module):
    def __init__(self, R: np.ndarray, W: np.ndarray, b: np.ndarray):
        super().__init__()

        R = np.asarray(R)
        W = np.asarray(W)
        b = np.asarray(b)

        self.conv0 = spectal_convolution_from_nola_weights(R)

        # Linear skip connection
        D = R.ndim - 2

        assert W.shape == (self.conv0.out_channels, self.conv0.in_channels)
        assert b.shape == (self.conv0.out_channels,)

        if D == 1:
            Conv = nn.Conv1d
        elif D == 2:
            Conv = nn.Conv2d
        else:
            assert D == 3
            Conv = nn.Conv3d

        self.w0 = Conv(self.conv0.in_channels, self.conv0.out_channels, 1)

        self.w0.weight.data = torch.tensor(
            np.expand_dims(W, axis=tuple(range(2, 2 + D)))
        )
        self.w0.bias.data = torch.tensor(b)

    def forward(self, x):
        # copied from https://github.com/pdebench/PDEBench/blob/7acb9945dfb7d714e2cfd1b7ec3eb2db2c20e6c8/pdebench/models/fno/fno.py#L111-L113
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        return x


def skip_if_case_unsupported(
    modes_shape: tuple[int, ...],
    output_grid_shape: tuple[int, ...] | None,
) -> None:
    if any(n % 2 != 0 for n in modes_shape[:-1]):
        pytest.skip("PDEBench does not support odd number of modes.")

    if output_grid_shape is not None:
        pytest.skip("PDEBench FNO block does not support output interpolation.")


########################################################################################
# copied from https://github.com/pdebench/PDEBench/blob/7acb9945dfb7d714e2cfd1b7ec3eb2db2c20e6c8/pdebench/models/fno/fno.py
########################################################################################

# fmt: off

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
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

# fmt: on
