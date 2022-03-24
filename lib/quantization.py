import torch
import torch.autograd.Function as F

class BlockQuantization3dFunc(F):
    @staticmethod
    def forward(ctx, input, alpha, bitwidth, symmetric = False):
        D, Cb1, b1, Hb2, b2, Wb3, b3 = input.shape
        assert(input.is_contiguous())
        assert(alpha.ndim == 3 and alpha.shape[0] == b1 and alpha.shape[1] == b2 and alpha.shape[2] == b3)

        if not symmetric:
            quant_min = 0
            quant_max = 2 ** bitwidth - 1
            input_flat = input.flatten(1)
            input_min, _ = input_flat.min(dim = 1)
            shifted_input = input_flat - input_min.view(-1, 1)
            input = shifted_input.view(*input.shape)
            shift = input_min.view(-1, 1, 1, 1, 1, 1, 1)

        else:
            quant_min = - (2 ** (bitwidth - 1))
            quant_max = - quant_min - 1
            shift = 0

        # Quantization and de-quantization
        output = input / alpha.view(1, 1, b1, 1, b2, 1, b3)
        output = output.round().clamp(quant_min, quant_max)
        output = output * alpha.view(1, 1, b1, 1, b2, 1, b3) + shift

        return output, shift

    @staticmethod
    def backward(ctx, grad_output, grad_shift):
        return grad_output, None, None, None
