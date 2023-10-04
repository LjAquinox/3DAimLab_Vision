import cv2
import numpy as np


def is_in_vicinity(color, target_color=(255, 120, 55), tolerance=30):
    """
    Check if a color is in the vicinity of a target color.
    """
    return np.linalg.norm(np.array(color) - np.array(target_color)) < tolerance

def resize_image(img, factor=0.25):
    """
    Resize the given image by a factor.

    Parameters:
    - img (numpy array): Image data.
    - factor (float): Factor by which to resize the image. Default is 0.25.

    Returns:
    - Resized image.
    """
    new_dimensions = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    resized_img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_img


def process_image(img):
    """
    Process the image based on the given conditions.

    Parameters:
    - img (numpy array): Image data.

    Returns:
    - Processed image.
    """
    # Convert image color space from BGR to RGB
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img
    # Create a mask based on the vicinity to the orange color
    mask = np.zeros_like(img_rgb)
    for y in range(img_rgb.shape[0]):
        for x in range(img_rgb.shape[1]):
            if is_in_vicinity(img_rgb[y, x]):
                mask[y, x] = img_rgb[y, x]


    # Apply the neighborhood check to refine the mask
    output_img = np.zeros_like(mask)
    kernel_size = 7
    offset = kernel_size // 2
    for y in range(offset, mask.shape[0] - offset):
        for x in range(offset, mask.shape[1] - offset):
            if not np.array_equal(mask[y, x], [0, 0, 0]):
                neighborhood = mask[y - offset:y + offset + 1, x - offset:x + offset + 1]
                non_black_pixels = np.sum(np.any(neighborhood != [0, 0, 0], axis=2))
                if non_black_pixels > (kernel_size * kernel_size / 2):
                    output_img[y, x] = img_rgb[y, x]

    return output_img

# Example usage:
# img = cv2.imread(".\Images2\my_screenshot0.png")
# processed_img = process_image(img)
