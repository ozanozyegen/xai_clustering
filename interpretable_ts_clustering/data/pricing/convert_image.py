import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
# from skimage import color

from typing import Union


def arr2imagearr(arr: Union[list, np.ndarray], shape):
    plt.ioff()
    fig = plt.gcf()

    DPI = fig.get_dpi()
    fig = plt.figure(figsize=(shape[0]/DPI, shape[1]/DPI), dpi=DPI, frameon=False)
    fig.set_size_inches(shape[0]/float(DPI),shape[1]/float(DPI))

    ax = fig.gca()
    canvas = FigureCanvas(fig)

    ax.set_axis_off()
    fig.add_axes(ax)

    plt.plot(arr)

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.canvas.draw()

    data = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    gray_image = color.rgb2gray(data)
    return gray_image
