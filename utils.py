import numpy as np
import cv2


def equalize_hist(channel, bins):
    """
    Based on https://docs.opencv.org/4.9.0/d4/d1b/tutorial_histogram_equalization.html

    :param channel: The channel to be equalized
    :param bins: The number of bins to be used in the histogram (number of possible values for the channel)
    """
    hist = np.zeros(bins)
    for val in channel.flatten():
        hist[val] += 1

    # Normalize the histogram so that the sum of all bins is 1
    hist = hist / (channel.shape[0] * channel.shape[1])

    # Create the cumulative distribution function
    cdf = np.zeros(bins)
    cdf[0] = hist[0]

    for i in range(1, bins):
        cdf[i] = cdf[i - 1] + hist[i]

    # Normalize the CDF so that the maximum value is
    # the maximum possible value for the channel
    cdf = cdf * (bins - 1)

    equalized_channel = np.zeros(channel.shape, dtype=np.uint8)

    # Map the original values to the ones in the CDF
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
    # Normalize the channel so that the maximum value is 1
    channel = channel / max_value

    # Apply the gamma correction
    corrected_channel = np.power(channel, gamma)

    # Change the channel back to the original range
    corrected_channel = corrected_channel * max_value

    return corrected_channel.astype(np.uint8)


def correct_images(images, gamma, max_vals):
    images_eq = []
    images_eq_gamma_corrected = []

    for img in images:
        channels = cv2.split(img)

        img_equalized = []
        img_eq_gamma_corrected = []

        for channel, max_val in zip(channels, max_vals):
            ch_equalized = equalize_hist(channel, max_val + 1)
            ch_eq_gamma_corrected = gamma_correction(ch_equalized, gamma, max_val)

            img_equalized.append(ch_equalized)
            img_eq_gamma_corrected.append(ch_eq_gamma_corrected)

        img_equalized = cv2.merge(img_equalized)
        img_eq_gamma_corrected = cv2.merge(img_eq_gamma_corrected)

        images_eq.append(img_equalized)
        images_eq_gamma_corrected.append(img_eq_gamma_corrected)

    return images_eq, images_eq_gamma_corrected


def get_cdf(channel, bins=256):
    hist = np.zeros(bins)
    for val in channel.flatten():
        hist[val] += 1
    hist = hist / (channel.shape[0] * channel.shape[1])

    cdf = np.zeros(bins)
    cdf[0] = hist[0]

    for i in range(1, bins):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf = cdf * (bins - 1)

    return cdf.astype(np.uint8)


def change_images_cdf(images, target_cdfs):
    images_changed_cdf = []

    for img in images:
        channels = cv2.split(img)

        img_changed_cdf = []

        for channel, target_cdf in zip(channels, target_cdfs):
            channel_cdf = get_cdf(channel, bins=target_cdf.shape[0])
            # interpolate between the original CDF and the target CDF
            mapping = np.interp(
                channel_cdf, target_cdf, np.arange(target_cdf.shape[0])
            ).astype(np.uint8)
            new_channel = mapping[channel]
            img_changed_cdf.append(new_channel)

        img_changed_cdf = cv2.merge(img_changed_cdf)
        images_changed_cdf.append(img_changed_cdf)

    return images_changed_cdf


def convert_images(images, cvt_type):
    new_images = []
    for img in images:
        new_images.append(cv2.cvtColor(img, cvt_type))
    return np.array(new_images)
