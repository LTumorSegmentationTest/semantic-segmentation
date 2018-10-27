#------------------------------------------------------
# Writed by: Zhengkai Jiang
# Combining local and global self-attention for semantic segmentation
#------------------------------------------------------

"""Synchronized Cross-GPU Batch Normalization functions"""
import torch
from torch.autograd import Variable, Function
from .. import ops

__all__ = ['sum_square', 'batchnormtrain']

def sum_square(input):
    r"""Calculate sum of elements and sum of squares for Batch Normalization"""
    return _sum_square.apply(input)


class _sum_square(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input.is_cuda:
            xsum, xsqusum = ops.gpu.sumsquare_forward(input)
        else:
            xsum, xsqusum = ops.cpu.sumsquare_forward(input)
        return xsum, xsqusum

    @staticmethod
    def backward(ctx, gradSum, gradSquare):
        input, = ctx.saved_variables
        if input.is_cuda:
            gradInput = ops.gpu.sumsquare_backward(input, gradSum, gradSquare)
        else:
            raise NotImplemented
        return gradInput


class _batchnormtrain(Function):
    @staticmethod
    def forward(ctx, input, mean, std, gamma, beta):
        ctx.save_for_backward(input, mean, std, gamma, beta)
        if input.is_cuda:
            output = ops.gpu.batchnorm_forward(input, mean, std, gamma, beta)
        else:
            output = ops.cpu.batchnorm_forward(input, mean, std, gamma, beta)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        input, mean, std, gamma, beta = ctx.saved_variables
        if gradOutput.is_cuda:
            gradInput, gradMean, gradStd, gradGamma, gradBeta = \
                ops.gpu.batchnorm_backward(gradOutput, input, mean,
                                           std, gamma, beta, True)
        else:
            raise NotImplemented
        return gradInput, gradMean, gradStd, gradGamma, gradBeta


def batchnormtrain(input, mean, std, gamma, beta):
    r"""Applies Batch Normalization over a 3d input that is seen as a
    mini-batch.
    .. _encoding.batchnormtrain:
    .. math::
        y = \frac{x - \mu[x]}{ \sqrt{var[x] + \epsilon}} * \gamma + \beta
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """
    return _batchnormtrain.apply(input, mean, std, gamma, beta)