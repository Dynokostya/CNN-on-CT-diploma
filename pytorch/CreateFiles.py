import os


class ProjectFoldersCreator:
    def __init__(self, base_path, num_patients):
        self.base_path = base_path
        self.num_patients = num_patients

    def create_folders(self):
        for i in range(0, self.num_patients):
            patient_folder = os.path.join(self.base_path, f'I_00{i:02d}')
            dicom_folder = os.path.join(patient_folder, 'DICOM')
            results_folder = os.path.join(patient_folder, 'DICOMCUT')
            png_folder = os.path.join(patient_folder, 'DICOMTIFF')
            for folder in [patient_folder, dicom_folder, results_folder, png_folder]:
                os.makedirs(folder, exist_ok=True)

            # tiff_folder = os.path.join(patient_folder, 'DICOMTIFF')
            # if os.path.exists(png_folder):
            #     os.rename(png_folder, tiff_folder)


# Example usage
if __name__ == "__main__":
    path = 'PatientData'
    patients_amount = 16
    folder_creator = ProjectFoldersCreator(path, patients_amount)
    folder_creator.create_folders()

    # Rename dicomm files
    # patient_folder = os.path.join(path, f'I_0009')
    # dicom_folder = os.path.join(patient_folder, 'DICOM')
    # for file, i in zip(os.listdir(dicom_folder), range(363, 381)):
    #     os.rename(os.path.join(dicom_folder, file), os.path.join(dicom_folder, f'I00{i:02d}'))
