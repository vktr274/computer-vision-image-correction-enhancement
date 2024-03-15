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


def gamma_correction(img, gamma, max_values):
    """
    Based on https://docs.opencv.org/4.9.0/d3/dc1/tutorial_basic_linear_transform.html

    :param img: The image to be corrected.
    :param gamma: The gamma value. When gamma < 1, the original dark
    regions will be brighter and the histogram will be shifted to
    the right whereas it will be the opposite with gamma > 1.
    :param max_values: The maximum value of each channel of the image.
    E.g. [255, 255, 255] for RGB images or [180, 255, 255] for HSV images.
    """
    split_img = cv2.split(img)
    corrected_img = []

    for channel, max_value in zip(split_img, max_values):
        # Normalize the channel so that the maximum value is 1
        corrected_channel = channel / max_value
        # Apply the gamma correction
        corrected_channel = np.power(corrected_channel, gamma)
        # Change the channel back to the original range
        corrected_channel = corrected_channel * max_value
        corrected_img.append(corrected_channel.astype(np.uint8))

    return cv2.merge(corrected_img)


def correct_images(images, max_vals):
    """
    Correct the images in an array using histogram equalization.

    :param images: The images to be corrected.
    :param max_vals: The maximum value of each channel of the images.
    E.g. [255, 255, 255] for RGB images or [180, 255, 255] for HSV images.
    """
    images_eq = []

    for img in images:
        channels = cv2.split(img)

        img_equalized = []

        for channel, max_val in zip(channels, max_vals):
            ch_equalized = equalize_hist(channel, max_val + 1)
            img_equalized.append(ch_equalized)

        img_equalized = cv2.merge(img_equalized)

        images_eq.append(img_equalized)

    return images_eq


def get_cdf(channel, bins=256):
    """
    Calculate the cumulative distribution function (CDF) of a channel.

    :param channel: The channel to get the CDF from.
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

    return cdf.astype(np.uint8)


def change_images_cdf(images, target_cdfs):
    """
    Change the CDF of the images to match the target CDFs.

    :param images: The images to be changed.
    :param target_cdfs: The target CDFs to be matched. E.g. [cdf1, cdf2, cdf3] for 3 channel images.
    """
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
    """
    Helper function to convert multiple images to a different color space.
    """
    new_images = []
    for img in images:
        new_images.append(cv2.cvtColor(img, cvt_type))
    return np.array(new_images)
