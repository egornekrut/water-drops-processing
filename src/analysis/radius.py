from typing import Tuple

import numpy as np
from PIL import Image


def find_center_of_mass(img_array: np.ndarray) -> Tuple[int, int]:
    # Find the indices where red pixels are present
    mask_pixels = np.where(img_array == True)

    # Calculate the center of mass
    center_x = int(np.round(np.mean(mask_pixels[1])))
    center_y = int(np.round(np.mean(mask_pixels[0])))

    return center_x, center_y

def ray_diameter_estimator(img_mask: np.ndarray, n_radius: int = 16) -> Tuple[float, float, float]:
    if len(img_mask.shape) > 2:
        raise ValueError

    img_mask = img_mask.astype(bool)
    img_mask_pil = Image.fromarray(img_mask.astype('uint8') * 255).convert('L')

    c_mass_x, c_mass_y = find_center_of_mass(img_mask)

    # Get image dimensions
    width, height = img_mask_pil.size

    # Store the lengths of the segments
    segment_lengths = []

    # Draw segments from the center to the border of the red area
    for angle in np.linspace(0, 2 * np.pi, n_radius, endpoint=False):  # Divide the circle into 16 equal parts
        # Initialize segment parameters
        x, y = c_mass_x, c_mass_y
        step = 1  # Adjust the step size for finer resolution

        # Move along the segment until a black pixel is encountered or the border of the image is reached
        while 0 <= x < width and 0 <= y < height and img_mask_pil.getpixel((x, y)) != 0:
            x = int(c_mass_x + step * np.cos(angle))
            y = int(c_mass_y + step * np.sin(angle))
            step += 1

        # Calculate the length of the segment and store it
        segment_length = np.sqrt((x - c_mass_x)**2 + (y - c_mass_y)**2)
        segment_lengths.append(segment_length)

    return np.mean(segment_lengths) * 2, c_mass_x, c_mass_y
