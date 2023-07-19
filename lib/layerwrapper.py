import torch
import torch.nn as nn

# Define WrappedGPT class
from torch import nn as nn


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", activation_strength_metric="norm"):
        self.activation_strength_metric = activation_strength_metric
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        else:
            print(f'WARNGING dfiferent layer tpye {type(self.layer)}')
            print(f'WARNGING dfiferent layer tpye {type(self.layer)}')
            print(f'WARNGING dfiferent layer tpye {type(self.layer)}')

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        if self.activation_strength_metric == "norm":
            scaler = torch.norm(inp, p=2, dim=1)
        if self.activation_strength_metric == "var":
            scaler = torch.std(inp, dim=1)
        if self.activation_strength_metric == "percentile":
            scaler = torch.mean(inp, dim=1) + 2 * torch.std(inp, dim=1)
        self.scaler_row += scaler ** 2 / self.nsamples


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sparsity_ratio):
        sorted_mask = input.sort(stable=True)[1]
        smallest_indices = sorted_mask[:int(input.shape[0] * sparsity_ratio)]
        mask = torch.ones_like(input, dtype=input.dtype)
        mask[smallest_indices] = 0.
        # todo can try have the gradient be mask * input
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # todo maybe just return grad_output??
        return grad_output * mask, None


class Binarize_ST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sparsity_ratio):
        sorted_mask = input.sort(stable=True)[1]
        smallest_indices = sorted_mask[:int(input.shape[0] * sparsity_ratio)]
        mask = torch.ones_like(input, dtype=input.dtype)
        mask[smallest_indices] = 0.
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class Binarize_Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sparsity_ratio):
        return torch.bernoulli(torch.sigmoid(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Binarize_Sigmoid_ST(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sparsity_ratio):
        mask = torch.bernoulli(torch.sigmoid(input))
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask, None


class Wrapper(nn.Module):

    def __init__(self, layer, args, track, layer_id=0, layer_name="none"):
        super(Wrapper, self).__init__()
        self.args = args
        self.layer_name = layer_name
        self.track = track
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.scaler_out = torch.zeros((self.rows), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name
        # init random with mean 1 and small std.
        self.mask = torch.nn.Parameter(torch.randn(self.layer.in_features, device=self.dev) * 0.01 + 1)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        assert isinstance(self.layer, nn.Linear)
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        if self.layer_name == 'q_proj':
            out = out.squeeze(0).t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.scaler_out *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp

        inp = inp.type(torch.float32)
        out = out.type(torch.float32)
        scaler = torch.norm(inp, p=2, dim=1)
        scaler_out = torch.norm(out, p=2, dim=1)

        self.scaler_row += scaler ** 2 / self.nsamples
        # self.scaler_out += scaler_out ** 2 / self.nsamples

    def forward(self, x):
        if not self.track:
            if self.args.mask_binarizer == 'binarize':
                mask_binarizer = Binarize.apply
            elif self.args.mask_binarizer == 'binarize_st':
                mask_binarizer = Binarize_ST.apply

            mask = mask_binarizer(self.mask, self.args.sparsity_ratio).to(x)
            x = x * mask

        # Put your own logic here
        out = self.layer(x)

        if self.track:
            self.add_batch(x[0].data, out.data)
        return out

    def prune(self, args):

        outgoing_edges_norm = self.layer.weight.data.norm(p=1, dim=0) / self.layer.weight.data.shape[0]
        average_logits = torch.sqrt(self.scaler_row)  # not necesserly need sqrt
        #
        scores = average_logits * outgoing_edges_norm  # this should be after the relu
        self.mask.data = scores
        self.args = args
