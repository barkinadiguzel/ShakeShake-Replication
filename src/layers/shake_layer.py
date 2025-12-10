import torch


class ShakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, training):
        ctx.training = training
        ctx.save_for_backward(x1, x2)
        if training:
            b = x1.size(0)
            alpha = torch.rand(b,1,1,1, device=x1.device)
            ctx.alpha_fwd = alpha
            return alpha * x1 + (1 - alpha) * x2
        else:
            return 0.5 * x1 + 0.5 * x2

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2 = ctx.saved_tensors
        if ctx.training:
            b = grad_output.size(0)
            alpha = torch.rand(b,1,1,1, device=grad_output.device)
            grad_x1 = alpha * grad_output
            grad_x2 = (1 - alpha) * grad_output
        else:
            grad_x1 = grad_output * 0.5
            grad_x2 = grad_output * 0.5
        return grad_x1, grad_x2, None


def shake_shake(x1, x2, training):
    return ShakeFunction.apply(x1, x2, training)
