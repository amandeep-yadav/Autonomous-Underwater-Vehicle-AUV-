import cv2
import numpy as np
import os

# Helper function to create the output directory
def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Enhancement Functions
def reduce_hue(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * 0.5  # Reduce hue
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def increase_brightness(image):
    return cv2.convertScaleAbs(image, alpha=1.0, beta=50)  # Increase brightness

def increase_exposure(image):
    return cv2.convertScaleAbs(image, alpha=1.5, beta=0)  # Increase exposure by increasing pixel intensity

def add_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Increase saturation
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Main Enhancement Pipeline
def enhance_and_save(image_path, label, output_dir, enhancements):
    """
    Applies chosen enhancements on the given image and saves the enhanced images and labels.

    :param image_path: Path to the input image.
    :param label: Original label for the input image.
    :param output_dir: Directory to save enhanced images.
    :param enhancements: List of enhancement functions to apply.
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

    # Apply each enhancement
    for enhancement in enhancements:
        enhanced_image = enhancement(image)
        enhanced_image_path = os.path.join(output_dir, f"{image_counter}.jpg")
        cv2.imwrite(enhanced_image_path, enhanced_image)
        with open(os.path.join(output_dir, f"{image_counter}.txt"), 'w') as f:
            f.write(label)  # The label remains unchanged
        image_counter += 1

# Example Usage
if __name__ == "__main__":
    image_path = "/kaggle/input/validation-set/validation/images/frame113_jpg.rf.901f02579c16489fcb9b47d0b05d5913.jpg"  # Path to input image
    label = "/kaggle/input/validation-set/validation/labels/frame113_jpg.rf.901f02579c16489fcb9b47d0b05d5913.txt"  # Example label (e.g., YOLO format)
    output_dir = "enhanced_results"  # Directory to save enhanced images

    # Select enhancements to apply
    enhancements = [
        reduce_hue,         # Reduce hue
        increase_brightness,  # Increase brightness
        increase_exposure,    # Increase exposure
        add_saturation       # Add saturation
    ]

    enhance_and_save(image_path, label, output_dir, enhancements)
