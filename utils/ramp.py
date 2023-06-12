import numpy as np


def no_ramp():
    """no rampup"""
    def warpper(epoch):
        return 1.0
    return warpper

def pseudo_rampup(T1, T2):
    def warpper(epoch):
        if epoch > T1:
            alpha = (epoch-T1) / (T2-T1)
            if epoch > T2:
                alpha = 1.0
        else:
            alpha = 0.0
        return alpha
    return warpper

def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if rampup_length == 0:
            return 1.0
        elif epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


def linear_rampup(rampup_length):
    """Linear rampup"""
    def warpper(epoch):
        if rampup_length == 0:
            return 1.0
        elif epoch < rampup_length:
            return float(np.clip(epoch / rampup_length, 0.0, 1.0))
        else:
            return 1.0
    return warpper
