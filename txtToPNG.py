import numpy as np
import cv2

def read_txt_image(file_path):
    # Read the text file and split into real and imaginary parts
    with open(file_path, 'r') as file:
        data = file.read().split()
    
    real_part = []
    
    for d in data:
        try:
            # Extract the real part of the complex number
            real_str = d.split(',')[0][1:]  
            real_value = int(float(real_str))  
            real_part.append(real_value)
        except ValueError as e:
            print(f"Skipping invalid entry '{d}': {e}")
    
    return np.array(real_part)

def convert_to_image(data, size):
    # Normalize the data to the 0-255 range if necessary
    normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    
    # Reshape the data to the desired size
    image = normalized_data.reshape(size).astype(np.uint8)
    
    return image

def save_image(image, output_path):
    # Save the image as a PNG file
    cv2.imwrite(output_path, image)

# File paths
input_file = 'fft_txt/1024_fft.txt'
output_file = 'fft_png/1024_fft.png'
image_size = (1024, 1024)

# Read the image data from the text file
real_part = read_txt_image(input_file)

# Ensure the data length matches the expected size
if real_part.size != image_size[0] * image_size[1]:
    raise ValueError(f"Data size {real_part.size} does not match expected image size {image_size[0] * image_size[1]}")

# Convert the real part to an image
image = convert_to_image(real_part, image_size)

# Save the image as a PNG file
save_image(image, output_file)


