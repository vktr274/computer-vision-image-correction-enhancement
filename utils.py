import numpy as np


def equalize_hist(channel, bins):
    """
    Based on https://docs.opencv.org/4.9.0/d4/d1b/tutorial_histogram_equalization.html

    :param channel: The channel to be equalized
    :param bins: The number of bins to be used in the histogram (number of possible values for the channel)
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


def gamma_correction(channel, gamma, max_value):
    """
    Based on https://docs.opencv.org/4.9.0/d3/dc1/tutorial_basic_linear_transform.html

    :param channel: The channel to be corrected
    :param gamma: The gamma value. When gamma < 1, the original dark
    regions will be brighter and the histogram will be shifted to
    the right whereas it will be the opposite with gamma > 1.
    :param max_value: The maximum value of the channel
    """
    channel = channel / max_value

    corrected_channel = np.power(channel, gamma)
    corrected_channel = corrected_channel * max_value

    return corrected_channel.astype(np.uint8)
