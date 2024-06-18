import os
import random


def copy_file(src, dst):
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # Copy the file
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            fdst.write(fsrc.read())


def split_dataset(folder_path, train_ratio=0.8):
    # Ensure the train and test directories exist
    train_folder = os.path.join(folder_path, 'train')
    test_folder = os.path.join(folder_path, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get all TSV files in the folder
    tsv_files = [f for f in os.listdir(folder_path) if f.endswith('.tsv')
                 and os.path.isfile(os.path.join(folder_path, f))]

    # Shuffle the files to ensure randomness
    random.shuffle(tsv_files)

    # Calculate the split index
    split_index = int(len(tsv_files) * train_ratio)

    # Split the files
    train_files = tsv_files[:split_index]
    test_files = tsv_files[split_index:]

    # Copy the files to the respective directories
    for f in train_files:
        copy_file(os.path.join(folder_path, f), os.path.join(train_folder, f))

    for f in test_files:
        copy_file(os.path.join(folder_path, f), os.path.join(test_folder, f))

    print(f"Copied {len(train_files)} files to {train_folder}")
    print(f"Copied {len(test_files)} files to {test_folder}")


# Before running this script, manually split the "cleanedFiles" folder
# into "cleanedBeethovenFiles" and "cleanedMozartFiles".
# Use the following file path for Beethoven.
folder_path = 'outputFiles/cleanedFiles/cleanedBeethovenFiles'
# Use the following file path for Mozart.
# folder_path = 'outputFiles/cleanedFiles/cleanedMozartFiles'
split_dataset(folder_path)
