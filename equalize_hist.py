import numpy as np


def equalize_hist(channel, bins):
    """
    Based on https://docs.opencv.org/4.9.0/d4/d1b/tutorial_histogram_equalization.html
    """
    hist = np.zeros(bins)
    for val in channel.flatten():
        hist[val] += 1
    hist = hist / (channel.shape[0] * channel.shape[1])

    cdf = np.zeros(bins)
    cdf[0] = hist[0]

    for i in range(1, bins):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf = cdf * (bins - 1)

    equalized_channel = np.zeros(channel.shape, dtype=np.uint8)
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            equalized_channel[i, j] = cdf[channel[i, j]]

    return equalized_channel
