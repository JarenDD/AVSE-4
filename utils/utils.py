import sys
import os

import scipy, math
import numpy as np
import torch

EPS = 1e-8

def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **kwargs, **config['args'])

SPEECH_FILTER = scipy.io.loadmat(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "speech_weight.mat"
    ),
    squeeze_me=True,
)
SPEECH_FILTER = np.array(SPEECH_FILTER["filt"])


def apply_ramp(y, fs=16000, dur=0.5):
    """Apply half cosine ramp into and out of signal

    dur - ramp duration in seconds
    """
    ramp = np.cos(np.linspace(math.pi, 2 * math.pi, int(fs * dur)))
    ramp = (ramp + 1) / 2
    # y = np.array(x)
    y[0 : len(ramp)] *= ramp
    y[-len(ramp) :] *= ramp[::-1]
    return y


