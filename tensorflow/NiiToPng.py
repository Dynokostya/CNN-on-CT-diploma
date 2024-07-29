import nibabel as nib
from PIL import Image
import os
import numpy as np


def save_nii_slices_as_png(nii_path, output_dir, num):
    # Load the NIfTI file
    nii = nib.load(nii_path)
    data = nii.get_fdata()

    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each slice
    # len = data.shape[2]
    for i in range(data.shape[2] - 1, -1, -1):
        # Extract the slice
        slice_data = data[:, :, i]

        # Check if the slice contains any relevant data (not completely black)
        if np.max(slice_data) > 0:  # Only process slices with information
            # Normalize and convert to 8-bit
            slice_normalized = (np.interp(slice_data, (slice_data.min(), slice_data.max()), (0, 255))).astype(np.uint8)

            # Convert to image
            slice_image = Image.fromarray(slice_normalized)
            slice_image = slice_image.rotate(270, expand=True)
            slice_image = slice_image.transpose(Image.FLIP_LEFT_RIGHT)
            # Save the slice as PNG
            slice_image.save(os.path.join(output_dir, f'1_00{num:02d}_mask_{data.shape[2] - i:03}.png'))


# Example usage (commented out to adhere to instructions)
num = 15
nii_file = f'PatientData/I_00{num:02d}/mask.nii'
output_path = f'PatientData/I_00{num:02d}/Masks'

save_nii_slices_as_png(nii_file, output_path, num)
