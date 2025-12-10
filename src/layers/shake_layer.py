import torch
from torch.autograd import Function

class ShakeFn(Function):
    @staticmethod
    def forward(ctx, x1, x2, training):
        ctx.training = training
        ctx.save_for_backward(x1, x2)
        if training:
            b = x1.size(0)
            a = torch.rand(b,1,1,1, device=x1.device)
            ctx.a_f = a
            return a*x1 + (1-a)*x2
        return 0.5*(x1+x2)

    @staticmethod
    def backward(ctx, g):
        x1,x2 = ctx.saved_tensors
        if ctx.training:
            b = g.size(0)
            a = torch.rand(b,1,1,1, device=g.device)
            gx1 = a*g
            gx2 = (1-a)*g
            return gx1,gx2,None
        return 0.5*g,0.5*g,None

def shake_forward(x1,x2,training):
    return ShakeFn.apply(x1,x2,training)
