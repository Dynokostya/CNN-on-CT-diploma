import os

def add_underscore_to_filenames(directory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            name, ext = os.path.splitext(filename)
            new_filename = name + "_" + ext
            os.rename(
                os.path.join(directory, filename),
                os.path.join(directory, new_filename)
            )

# Replace 'DIR' with the path to your directory
directory_path = 'C:/Users/kosty/Desktop/PatientData/I_0013/DICOMCUT'
add_underscore_to_filenames(directory_path)
