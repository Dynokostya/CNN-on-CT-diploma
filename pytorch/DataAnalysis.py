import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, mode
from scipy.stats import entropy


def load_dataset(base_dir):
    all_images = []
    all_labels = []

    for i, patient in enumerate(os.listdir(base_dir)[:16]):
        patient_path = os.path.join(base_dir, patient, 'DICOMCUT')
        patient_images = os.listdir(patient_path)
        if patient_images:
            patient_image_arrays = []
            for img_file in patient_images:
                img = Image.open(os.path.join(patient_path, img_file))
                img_array = np.array(img)
                patient_image_arrays.append(img_array)
            all_images.append(patient_image_arrays)
            all_labels.append(patient)

    return all_images, all_labels


def calculate_mean(images):
    return np.mean(images)


def calculate_median(images):
    return np.median(images)


def calculate_variance(images):
    return np.var(images)


def calculate_std(images):
    return np.std(images)


def calculate_min(images):
    return np.min(images)


def calculate_max(images):
    return np.max(images)


def calculate_iqr(images):
    q75, q25 = np.percentile(images, [75, 25])
    return q75 - q25


def calculate_skewness(images):
    return skew(images.flatten())


def calculate_kurtosis(images):
    return kurtosis(images.flatten())


def plot_histogram(data, labels, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(labels, data)
    plt.xlabel('Patients')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()


def main():
    # Path to your dataset
    base_dir = 'PatientData'

    # Load the dataset
    all_images, labels = load_dataset(base_dir)

    # Prepare metrics for visualization
    means = []
    medians = []
    variances = []
    std_devs = []
    mins = []
    maxs = []
    iqrs = []
    skewnesses = []

    # Iterate over each patient's data
    for i, images in enumerate(all_images):
        # Convert list of images to a single numpy array
        images_np = np.array(images)

        # Calculate metrics
        means.append(calculate_mean(images_np))
        medians.append(calculate_median(images_np))
        variances.append(calculate_variance(images_np))
        std_devs.append(calculate_std(images_np))
        mins.append(calculate_min(images_np))
        maxs.append(calculate_max(images_np))
        iqrs.append(calculate_iqr(images_np))
        skewnesses.append(calculate_skewness(images_np))

    # Plot histograms
    plot_histogram(means, labels, 'Mean Value per Patient', 'Mean Value')
    plot_histogram(medians, labels, 'Median Value per Patient', 'Median Value')
    plot_histogram(variances, labels, 'Variance per Patient', 'Variance')
    plot_histogram(std_devs, labels, 'Standard Deviation per Patient', 'Standard Deviation')
    plot_histogram(mins, labels, 'Min Value per Patient', 'Min Value')
    plot_histogram(maxs, labels, 'Max Value per Patient', 'Max Value')
    plot_histogram(iqrs, labels, 'Interquartile Range (IQR) per Patient', 'IQR')
    plot_histogram(skewnesses, labels, 'Skewness per Patient', 'Skewness')


if __name__ == "__main__":
    main()
