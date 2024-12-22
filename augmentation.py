import cv2
import numpy as np
import os
from skimage.util import random_noise

# Helper function to create the output directory
def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Augmentation Functions
def rotate_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def add_noise(image):
    noisy = random_noise(image, mode='s&p', amount=0.05)
    return np.array(255 * noisy, dtype='uint8')

def decrease_hue(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * 0.5  # Reduce hue
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def increase_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def increase_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def increase_brightness(image):
    return cv2.convertScaleAbs(image, alpha=1.2, beta=50)  # Increase brightness

# Main Augmentation Pipeline
def augment_and_save(image_path, label, output_dir, augmentations):
    """
    Applies chosen augmentations on the given image and saves the augmented images and labels.

    :param image_path: Path to the input image.
    :param label: Original label for the input image.
    :param output_dir: Directory to save augmented images.
    :param augmentations: List of augmentation functions to apply.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Create output directory if not exists
    create_output_dir(output_dir)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_counter = 0

    # Save the original image and label
    original_image_path = os.path.join(output_dir, f"{image_counter}.jpg")
    cv2.imwrite(original_image_path, image)
    with open(os.path.join(output_dir, f"{image_counter}.txt"), 'w') as f:
        f.write(label)
    image_counter += 1

    # Apply each augmentation
    for augmentation in augmentations:
        augmented_image = augmentation(image)
        augmented_image_path = os.path.join(output_dir, f"{image_counter}.jpg")
        cv2.imwrite(augmented_image_path, augmented_image)
        with open(os.path.join(output_dir, f"{image_counter}.txt"), 'w') as f:
            f.write(label)  # The label remains unchanged
        image_counter += 1

# Example Usage
if __name__ == "__main__":
    image_path = "/kaggle/input/validation-set/validation/images/frame113_jpg.rf.901f02579c16489fcb9b47d0b05d5913.jpg"  # Path to input image
    label = "/kaggle/input/validation-set/validation/labels/frame113_jpg.rf.901f02579c16489fcb9b47d0b05d5913.txt"  # Example label (e.g., YOLO format)
    output_dir = "augmented_results"  # Directory to save augmented images

    # Select augmentations to apply
    augmentations = [
        rotate_90,         # Rotate 90 degrees
        grayscale,         # Convert to grayscale
        add_noise,         # Add salt-and-pepper noise
        decrease_hue,      # Decrease hue
        increase_saturation,  # Increase saturation
        increase_contrast,    # Increase contrast
        increase_brightness   # Increase brightness
    ]

    augment_and_save(image_path, label, output_dir, augmentations)
