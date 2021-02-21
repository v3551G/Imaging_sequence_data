# -*- coding: utf-8 -*-
"""
@author: masterqkk, masterqkk@outlook.com
Environment:
    python: 3.6
    Pandas: 1.0.3
    matplotlib: 3.2.1
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyts.datasets import load_gunpoint
from pyts.image import RecurrencePlot

if __name__ == '__main__':
    X, _, _, _ = load_gunpoint(return_X_y=True)
    rp = RecurrencePlot(dimension=3, time_delay=3)
    X_new = rp.transform(X)
    rp2 = RecurrencePlot(dimension=3, time_delay=10)
    X_new2 = rp2.transform(X)
    plt.figure()
    plt.suptitle('gunpoint_index_0')
    ax1 = plt.subplot(121)
    plt.imshow(X_new[0])
    plt.title('Recurrence plot, dimension=3, time_delay=3')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(cax=cax)

    ax1 = plt.subplot(122)
    plt.imshow(X_new2[0])
    plt.title('Recurrence plot, dimension=3, time_delay=10')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(cax=cax)
    plt.show()





