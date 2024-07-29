from PIL import Image
import os


def resize_images(directory, target_size=(512, 512)):
    """ Resize all PNG images in the specified directory to the target size. """
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                # Check if image is already at the target size
                if img.size != target_size:
                    # Resize the image using LANCZOS (formerly ANTIALIAS)
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    # Save the resized image back to the same location
                    resized_img.save(file_path)
                    print(f"Resized {filename}")


# Example usage
if __name__ == "__main__":
    directory = 'PatientData/I_0001/DICOMCUT'  # Replace with the path to your directory
    resize_images(directory)
