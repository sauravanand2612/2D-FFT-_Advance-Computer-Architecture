import numpy as np
import cv2

# Define a function to process images
def process_image(image_size):
    # Construct file paths
    input_file = f'source_rgb_image/{image_size}_rgb.png'
    output_file = f'source_gray_image/{image_size}_gray.txt'

    # Read the RGB image
    image = cv2.imread(input_file)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error loading image {input_file}")
        return

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image to a text file
    np.savetxt(output_file, gray_image, fmt='%d')

# List of image sizes to process
image_sizes = [512, 1024, 2048, 4096]

# Process each image in the list
for size in image_sizes:
    process_image(size)
