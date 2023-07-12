import torch
from einops import rearrange
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


def iou(x_gt, x, thres):
    thres_gt = 0.0

    # compute iou
    # > 0 free space, < 0 occupied
    x_gt_mask = x_gt.clone().detach()
    x_gt_mask[x_gt > thres_gt] = 0.0
    x_gt_mask[x_gt <= thres_gt] = 1.0

    x_mask = x.clone().detach()
    x_mask[x > thres] = 0.0
    x_mask[x <= thres] = 1.0

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    inter = rearrange(inter, "b c d h w -> b (c d h w)")
    union = rearrange(union, "b c d h w -> b (c d h w)")

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    return iou


#################### START: MISCELLANEOUS ####################
def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


#################### END: MISCELLANEOUS ####################


# Noam Learning rate schedule.
# From https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py
class NoamLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps**0.5 * min(
            last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]
