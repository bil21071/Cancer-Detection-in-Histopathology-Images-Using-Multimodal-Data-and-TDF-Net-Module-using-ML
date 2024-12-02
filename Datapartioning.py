import os
import openpyxl
import numpy as np

if __name__ == '__main__':

    # Set the dataset path
    data_root = r"C:\Users\Hp\Downloads\TDF_Net\TDF_Net\US3M\US3M\BD3M"
    excel_root = r"C:\Users\Hp\Downloads\TDF_Net\TDF_Net\US3M\US3M\BD3M.xlsx"

    # Get a list of subfolders
    dirs = os.listdir(data_root)

    # Load the Excel file
    workbook = openpyxl.load_workbook(excel_root)
    worksheet = workbook.active

    # Get the subfolder names and their corresponding labels
    folder_label = []
    for row in worksheet:
        # Check if the subfolder listed in the Excel is in the dataset
        if str(row[0].value) in dirs:
            dir_path = os.path.join(data_root, str(row[0].value))
            folder_label.append((dir_path, row[1].value))

    # Count the number of labels
    label_count = {0: 0, 1: 0}
    for _, label in folder_label:
        label_count[label] += 1

    # Division ratios
    total_samples = len(folder_label)
    train_ratio = 0.6  # Training set ratio
    valid_ratio = 0.2  # Validation set ratio
    test_ratio = 0.2  # Test set ratio

    # Calculate the number of samples for each dataset
    train_count = {0: int(train_ratio * label_count[0]), 1: int(train_ratio * label_count[1])}
    valid_count = {0: int(valid_ratio * label_count[0]), 1: int(valid_ratio * label_count[1])}
    test_count = {0: int(test_ratio * label_count[0]), 1: int(test_ratio * label_count[1])}

    # Calculate remainders for evenly adding insufficient samples
    remainder = {0: label_count[0] % 3, 1: label_count[1] % 3}

    # Shuffle the list of filenames
    np.random.shuffle(folder_label)

    # Create text files to store indices
    f_train = open(os.path.join(data_root, 'train.txt'), 'w')
    f_valid = open(os.path.join(data_root, 'valid.txt'), 'w')
    f_test = open(os.path.join(data_root, 'test.txt'), 'w')

    # Split into training, validation, and test sets
    train_samples = []
    valid_samples = []
    test_samples = []

    for filename, label in folder_label:
        # Assign samples based on the current label count
        if train_count[label] > 0:
            train_samples.append((filename, label))
            train_count[label] -= 1
        elif valid_count[label] > 0:
            valid_samples.append((filename, label))
            valid_count[label] -= 1
        elif test_count[label] > 0:
            test_samples.append((filename, label))
            test_count[label] -= 1
        else:
            # Use remainders to distribute any remaining samples
            if remainder[label] > 0:
                train_samples.append((filename, label))
                remainder[label] -= 1
            elif remainder[label] == 0:
                valid_samples.append((filename, label))
                remainder[label] -= 1
            else:
                test_samples.append((filename, label))
                remainder[label] -= 1

    # Shuffle the training, validation, and test sets
    np.random.shuffle(train_samples)
    np.random.shuffle(valid_samples)
    np.random.shuffle(test_samples)

    # Write to train.txt, valid.txt, test.txt
    for filename, label in train_samples:
        f_train.write(filename + '\t' + str(label) + '\n')

    for filename, label in valid_samples:
        f_valid.write(filename + '\t' + str(label) + '\n')

    for filename, label in test_samples:
        f_test.write(filename + '\t' + str(label) + '\n')

    # Close the files
    f_train.close()
    f_valid.close()
    f_test.close()

    print('Dataset partitioning successful!')