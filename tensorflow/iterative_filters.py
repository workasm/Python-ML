
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def xe(x,m):
    zz = (x - m) / (m/3)
    return np.exp(-zz*zz)

uu = scipy.io.loadmat('C:\work\ImageProcDocs\FastIterativeFilter\prefixed_double_filter.mat')

m = 204
signal = np.arange(1, 2*m+2, 1);

zsig = xe(signal,m)
zsig = zsig / sum(zsig)

#plt.plot(zsig)
filt = uu['MM'].flatten()

#plt.plot(filt)
