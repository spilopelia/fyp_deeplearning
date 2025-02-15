import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset

def load_numpy_files_generator(shards):
    # This generator takes a list of file paths (a shard) and yields examples.
    for file_paths in shards:
        for file_path in file_paths:
            # Load the numpy array and convert to a torch tensor.
            displacement_array = np.load(file_path)
            
            # Extract file id from the file name.
            file_name = os.path.basename(file_path)
            id = os.path.splitext(file_name)[0]
            
            yield {
                'displacement': displacement_array.astype(np.float32),
                'id': id,
            }

def main():
    csv_file_displacement = '/home/user/ckwan1/ml/displacement.csv'
    save_path = '/home/user/ckwan1/ml/huggingface_dataset_128_mul/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Read CSV to get the complete list of file paths.
    file_paths = pd.read_csv(csv_file_displacement)['file_path'].tolist()

    # Define the number of processes you'd like to use.
    num_proc = 32
    
    # Split file_paths into num_proc shards.
    # For example, if file_paths = [a, b, c, d, e, f, ...],
    # then shards will be: [[a, e, ...], [b, f, ...], [c, g, ...], [d, h, ...]]
    shards = [file_paths[i::num_proc] for i in range(num_proc)]
    
    # Use from_generator with gen_kwargs set to our shards, and set num_proc.
    dataset = Dataset.from_generator(
        load_numpy_files_generator,
        gen_kwargs={"shards": shards},
        num_proc=num_proc
    )
    
    dataset.save_to_disk(save_path)
    print("Dataset saved successfully.")

if __name__ == "__main__":
    main()
