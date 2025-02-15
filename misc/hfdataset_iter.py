import pandas as pd
import numpy as np
import torch
from datasets import IterableDataset
import os


def load_numpy_files_generator(csv_file_displacement):
    displacement_file_list = pd.read_csv(csv_file_displacement)
    displacement_file_paths = displacement_file_list['file_path'].tolist()

    for displacement_file_path in displacement_file_paths:
        displacement_array = np.load(displacement_file_path)  # Load NumPy array

        file_name = os.path.basename(displacement_file_path)  # Get file name
        id = os.path.splitext(file_name)[0]  # Remove file extension

        data_dict = {
            'displacement': displacement_array.astype(np.float32),  # Store as NumPy array
            'id': id,
        }
        yield data_dict


def main():
    csv_file_displacement = '/home/user/ckwan1/ml/displacement.csv'
    save_path = '/home/user/ckwan1/ml/huggingface_dataset_128_iter/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create an IterableDataset with tensors stored
    dataset = IterableDataset.from_generator(lambda: load_numpy_files_generator(csv_file_displacement))

    # Save dataset (HF Datasets will store tensors as NumPy arrays)
    dataset.save_to_disk(save_path)

    print("IterableDataset with tensors saved successfully.")


if __name__ == "__main__":
    main()
