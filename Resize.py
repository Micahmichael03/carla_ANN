#=============================================================================================================
# Resize-FILE-for-RGB-Detection-from-simulation-by-Michael_Micah,ENGR.DR.MARTINS_OBASEKI (2024)-----
#=============================================================================================================
from PIL import Image  # Import the Image module from the Pillow library, which provides image processing capabilities.
import os  # Import the os module, which provides functions for interacting with the operating system.

# Define a function to compress an image and reduce its file size.
def compress_image(input_path, output_path, target_size_kb, quality=85):
    # Open the image file from the input path.
    img = Image.open(input_path)
    
    # Save the image to the output path in JPEG format with the specified quality and optimization.
    img.save(output_path, format='JPEG', quality=quality, optimize=True)
    
    # Get the size of the saved image file in kilobytes.
    file_size_kb = os.path.getsize(output_path) / 1024
    
    # If the file size is still larger than the target size, reduce the quality incrementally.
    while file_size_kb > target_size_kb and quality > 10:
        quality -= 5  # Reduce the quality by 5.
        img.save(output_path, format='JPEG', quality=quality, optimize=True)  # Save the image again with the new quality setting.
        file_size_kb = os.path.getsize(output_path) / 1024  # Update the file size.
    
    return file_size_kb  # Return the final file size.

# Define the input folder containing the original images. Use a raw string literal to handle backslashes in the path.
input_folder = r'D:\Carla-data\rgb'

# Define the output folder where compressed images will be saved. Use a raw string literal to handle backslashes in the path.
output_folder = r'D:\Carla-data\compressed_rgb'

# Define the target file size for the compressed images in kilobytes.
target_size_kb = 150

# Check if the output folder exists, and create it if it doesn't.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over each file in the input folder.
for filename in os.listdir(input_folder):
    # Check if the file has a .png or .jpg extension.
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Construct the full input file path.
        input_path = os.path.join(input_folder, filename)
        
        # Construct the full output file path, changing the extension to .jpg.
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
        
        # Compress the image and get the final file size.
        file_size_kb = compress_image(input_path, output_path, target_size_kb)
        
        # Print a message indicating the filename and its final size.
        print(f"Compressed {filename} to {file_size_kb:.2f} KB")

# Print a message indicating that the compression process is completed.
print("Compression completed.")


"""
from PIL import Image: Imports the Image module from the Pillow library, which is used for opening, manipulating, and saving many different image file formats.
import os: Imports the os module, which provides a way of using operating system-dependent functionality like reading or writing to the file system.
compress_image: This function handles the compression of a single image. It tries to save the image with the specified quality and checks if the resulting file size is below the target size. If not, it reduces the quality incrementally until the file size is acceptable or the quality is too low.
input_folder: Specifies the directory where the original images are located.
output_folder: Specifies the directory where the compressed images will be saved. If this directory does not exist, it is created.
target_size_kb: Sets the desired maximum file size for the compressed images.
os.path.join: Joins one or more path components intelligently to form a valid path.
os.path.splitext: Splits the pathname into a pair (root, ext) such that root + ext == pathname, and ext is empty or begins with a period and contains at most one period.
"""