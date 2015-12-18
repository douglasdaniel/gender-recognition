#!/usr/bin/env python

import numpy as np
from sys import argv
from audacious import *
import matplotlib.pyplot as plt
import pylab

def findceps(x, n=None):

    def _unwrap(phase):
        samples = phase.shape[-1]
        unwrapped = np.unwrap(phase)
        center = (samples+1)//2
        if samples == 1: 
            center = 0  
        ndelay = np.array(np.round(unwrapped[...,center]/np.pi))
        unwrapped -= np.pi * ndelay[...,None] * np.arange(samples) / center
        return unwrapped, ndelay
        
    spectrum = np.fft.fft(x, n=n)
    unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
    log_spectrum = np.log(np.abs(spectrum)) + 1j*unwrapped_phase
    ceps = np.fft.ifft(log_spectrum).real

    return ceps, ndelay

sound = Speech(argv[1])

ceps, _ = findceps(sound.signal, 1000)

duration = 1.0
samples = int(sound.fs*duration)
t = np.arange(samples) / sound.fs
print t.shape
print ceps.shape
plt.plot(t[:ceps.shape[0]], ceps)
pylab.show()
