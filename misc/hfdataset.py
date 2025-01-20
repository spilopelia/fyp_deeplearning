import pandas as pd
import numpy as np
import torch
from datasets import Dataset
import os


def load_numpy_files_generator(csv_file_displacement,csv_file_density):
    displacement_file_list = pd.read_csv(csv_file_displacement)
    density_file_list = pd.read_csv(csv_file_density)
    displacement_file_paths = displacement_file_list['file_path'].tolist()   
    density_file_paths = density_file_list['file_path'].tolist()   
    for i in range(len(displacement_file_paths)):
        displacement_file_path = displacement_file_paths[i]
        displacement_array = np.load(displacement_file_path)
        displacement_tensor = torch.from_numpy(displacement_array)

        density_file_path = density_file_paths[i]
        density_array = np.load(density_file_path)
        density_tensor = torch.from_numpy(density_array)   

        file_name = os.path.basename(displacement_file_path)  # Get the file name
        id = os.path.splitext(file_name)[0]  # Remove the file extension

        data_dict = {
            'displacement': displacement_tensor,
            'density': density_tensor,
            'id': id,
        }
        
        yield data_dict

def main():
    csv_file_displacement = '/home/user/ckwan1/ml/displacement.csv' 
    csv_file_density = '/home/user/ckwan1/ml/density.csv' 

    save_path = '/home/user/ckwan1/ml/huggingface_dataset/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = Dataset.from_generator(lambda: load_numpy_files_generator(csv_file_displacement,csv_file_density))
    dataset.save_to_disk(save_path)

    print("Dataset saved successfully.")

if __name__ == "__main__":
    main()