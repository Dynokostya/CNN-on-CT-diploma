import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os


def load_image(path, mode='L'):
    """ Load an image and convert to specified mode. """
    with Image.open(path) as img:
        return img.convert(mode)


def apply_mask(reference_image, mask):
    """ Apply a binary mask to the reference image. """
    reference_array = np.array(reference_image)
    mask_array = np.array(mask)
    masked_image = np.where(mask_array > 128, reference_array, 0)  # Using a threshold of 128 for binary mask
    return Image.fromarray(masked_image.astype('uint8'))


def display_images(images):
    """ Display images using matplotlib. """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def process_all(tiff_files_dir, mask_files_dir, output_files_dir):
    """ Process all matching TIFF and mask files in specified directories. """
    tiff_files = sorted([f for f in os.listdir(tiff_files_dir)])
    mask_files = sorted([f for f in os.listdir(mask_files_dir)])
    for tiff_file, mask_file in zip(tiff_files, mask_files):
        tiff_image = load_image(os.path.join(tiff_files_dir, tiff_file))
        mask = load_image(os.path.join(mask_files_dir, mask_file), 'L')
        final_image = apply_mask(tiff_image, mask)
        final_image.save(os.path.join(output_files_dir, tiff_file.replace('TIFF.tif', 'CUT.png')))


# Example usage (uncomment for actual use)
num = 0
tiff_dir = f'PatientData/I_00{num:02d}/DICOMTIFF/'
mask_dir = f'PatientData/I_00{num:02d}/Masks/'
output_dir = f'PatientData/I_00{num:02d}/DICOMCUT/'
process_all(tiff_dir, mask_dir, output_dir)

#Не нужно сейчас писать код.
#Сейчас покажу иерархию снимков (уже готовых) для пациентов.
#1. Папка PatientData, в ней пациенты (папки) от I_0000 до I_0005. В каждом пациенте есть папка DICOMCUT, там находятся необходимые снимки. Нужно чтобы программа могла классифицировать снимки для каждого пациента. Для каждого пациента есть от 20 до 40 снимков. Разрешение каждого снимка (png) 512 на 512