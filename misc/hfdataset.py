import pandas as pd
import numpy as np
import torch
from datasets import Dataset
import os


def load_numpy_files_generator(csv_file_displacement):
    displacement_file_list = pd.read_csv(csv_file_displacement)
    displacement_file_paths = displacement_file_list['file_path'].tolist()   
    for i in range(len(displacement_file_paths)):
        displacement_file_path = displacement_file_paths[i]
        displacement_array = np.load(displacement_file_path)
        displacement_tensor = torch.from_numpy(displacement_array)

        file_name = os.path.basename(displacement_file_path)  # Get the file name
        id = os.path.splitext(file_name)[0]  # Remove the file extension

        data_dict = {
            'displacement': displacement_tensor.astype(np.float32),
            'id': id,
        }
        
        yield data_dict

def main():
    csv_file_displacement = '/home/user/ckwan1/ml/displacement.csv' 

    save_path = '/home/user/ckwan1/ml/huggingface_dataset_128/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = Dataset.from_generator(lambda: load_numpy_files_generator(csv_file_displacement))
    dataset.save_to_disk(save_path)

    print("Dataset saved successfully.")

if __name__ == "__main__":
    main()