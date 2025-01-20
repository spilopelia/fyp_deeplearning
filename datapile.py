import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import datasets
from datasets.distributed import split_dataset_by_node


def swap(array, i, j):
    array[[i, j]] = array[[j, i]]
    return array

class LoadRawDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        self.file_paths = self.load_csv(csv_file)
        self.augment = augment

    def load_csv(self, csv_file):
        csv_data = pd.read_csv(csv_file)
        
        if 'file_path' not in csv_data.columns:
            raise ValueError(f"'file_path' column not found in {csv_file}")
        
        file_paths = csv_data['file_path'].tolist()   
        return file_paths 

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        data = np.einsum('ijkl->lijk', data)
        LPT = data[6:9,:,:,:]  
        Nbody = data[0:3,:,:,:]  
        if self.augment:
            if np.random.rand() < .5:
                LPT = LPT[:, ::-1, ...]
                LPT[0] = -LPT[0]
                Nbody = Nbody[:, ::-1, ...]
                Nbody[0] = -Nbody[0]
            if np.random.rand() < .5:
                LPT = LPT[:, :, ::-1, ...]
                LPT[1] = -LPT[1]
                Nbody = Nbody[:, :, ::-1, ...]
                Nbody[1] = -Nbody[1]
            if np.random.rand() < .5:
                LPT = LPT[:, :, :, ::-1]
                LPT[2] = -LPT[2]
                Nbody = Nbody[:, :, :, ::-1]
                Nbody[2] = -Nbody[2]
            prand = np.random.rand()
            if prand < 1./6:
                LPT = np.transpose(LPT, axes=(0, 2, 3, 1))
                LPT = swap(LPT, 0, 2)
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 2, 3, 1))
                Nbody = swap(Nbody, 0, 2)
                Nbody = swap(Nbody, 0, 1)
            elif prand < 2./6:
                LPT = np.transpose(LPT, axes=(0, 2, 1, 3))
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 2, 1, 3))
                Nbody = swap(Nbody, 0, 1)
            elif prand < 3./6:
                LPT = np.transpose(LPT, axes=(0, 1, 3, 2))
                LPT = swap(LPT, 1, 2)
                Nbody = np.transpose(Nbody, axes=(0, 1, 3, 2))
                Nbody = swap(Nbody, 1, 2)
            elif prand < 4./6:
                LPT = np.transpose(LPT, axes=(0, 3, 1, 2))
                LPT = swap(LPT, 1, 2)
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 3, 1, 2))
                Nbody = swap(Nbody, 1, 2)
                Nbody = swap(Nbody, 0, 1)
            elif prand < 5./6:
                LPT = np.transpose(LPT, axes=(0, 3, 2, 1))
                LPT = swap(LPT, 0, 2)
                Nbody = np.transpose(Nbody, axes=(0, 3, 2, 1))
                Nbody = swap(Nbody, 0, 2)

        # Convert to PyTorch tensors
        LPT = torch.from_numpy(LPT.copy())
        Nbody = torch.from_numpy(Nbody.copy())

        return LPT, Nbody

class FastPMPile(L.LightningDataModule): 
    def __init__(
        self,
        train_csv_file: str = None,
        val_csv_file: str = None,
        test_csv_file: str = None,
        batch_size: int = 512,
        num_workers: int = 10,
        augment: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['kwargs'])

    def setup(self, stage):
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.batch_size = self.hparams.batch_size // dist.get_world_size()
        else:
            self.batch_size = self.hparams.batch_size

    def train_dataloader(self):
        train_dataset = LoadRawDataset(csv_file=self.hparams.train_csv_file, augment=self.hparams.augment)
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
            drop_last=True,
            #pin_memory=pin_memory,
        )

    def val_dataloader(self):
        val_dataset = LoadRawDataset(csv_file=self.hparams.train_csv_file, augment=False)
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            val_dataset,
            shuffle=False, 
            batch_size=self.batch_size,
            drop_last=False,  
            #pin_memory=pin_memory,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
        )
    
    def test_dataloader(self):
        test_dataset = LoadRawDataset(csv_file=self.hparams.train_csv_file, augment=False)
        pin_memory = torch.cuda.is_available()
        return DataLoader(
            test_dataset,
            shuffle=False,  
            batch_size=self.batch_size,
            drop_last=False,  
            #pin_memory=pin_memory,
            num_workers=self.hparams.num_workers,
            #persistent_workers=True,
        )

class AugmentedDataset(Dataset):
    def __init__(self, dataset, augment=False, density=True):
        self.dataset = dataset
        self.augment = augment
        self.density = density

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dset = self.dataset[idx]
        displacement = dset['displacement']
        displacement = np.einsum('ijkl->lijk', displacement)

        if self.density:
            density = dset['density']
            LPT = np.vstack((displacement[6:9], np.expand_dims(density[2], axis=0))).astype(np.float32)
            Nbody = np.vstack((displacement[0:3], np.expand_dims(density[0], axis=0))).astype(np.float32)
        else:
            LPT = displacement[6:9].astype(np.float32)
            Nbody = displacement[0:3].astype(np.float32)
        if self.augment:
            if np.random.rand() < .5:
                LPT = LPT[:, ::-1, ...]
                LPT[0] = -LPT[0]
                Nbody = Nbody[:, ::-1, ...]
                Nbody[0] = -Nbody[0]
            if np.random.rand() < .5:
                LPT = LPT[:, :, ::-1, ...]
                LPT[1] = -LPT[1]
                Nbody = Nbody[:, :, ::-1, ...]
                Nbody[1] = -Nbody[1]
            if np.random.rand() < .5:
                LPT = LPT[:, :, :, ::-1]
                LPT[2] = -LPT[2]
                Nbody = Nbody[:, :, :, ::-1]
                Nbody[2] = -Nbody[2]
            prand = np.random.rand()
            if prand < 1./6:
                LPT = np.transpose(LPT, axes=(0, 2, 3, 1))
                LPT = swap(LPT, 0, 2)
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 2, 3, 1))
                Nbody = swap(Nbody, 0, 2)
                Nbody = swap(Nbody, 0, 1)
            elif prand < 2./6:
                LPT = np.transpose(LPT, axes=(0, 2, 1, 3))
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 2, 1, 3))
                Nbody = swap(Nbody, 0, 1)
            elif prand < 3./6:
                LPT = np.transpose(LPT, axes=(0, 1, 3, 2))
                LPT = swap(LPT, 1, 2)
                Nbody = np.transpose(Nbody, axes=(0, 1, 3, 2))
                Nbody = swap(Nbody, 1, 2)
            elif prand < 4./6:
                LPT = np.transpose(LPT, axes=(0, 3, 1, 2))
                LPT = swap(LPT, 1, 2)
                LPT = swap(LPT, 0, 1)
                Nbody = np.transpose(Nbody, axes=(0, 3, 1, 2))
                Nbody = swap(Nbody, 1, 2)
                Nbody = swap(Nbody, 0, 1)
            elif prand < 5./6:
                LPT = np.transpose(LPT, axes=(0, 3, 2, 1))
                LPT = swap(LPT, 0, 2)
                Nbody = np.transpose(Nbody, axes=(0, 3, 2, 1))
                Nbody = swap(Nbody, 0, 2)

        return torch.from_numpy(LPT.copy()), torch.from_numpy(Nbody.copy())

class HuggingfaceLoader(L.LightningDataModule):  # use to load huggingface dataset
    def __init__(
        self,
        dataset_path: str,
        test_size: float = 0.1,
        shuffle: bool = True,
        batch_size: int = 512,
        num_workers: int = 10,
        augment: bool = True,
        density: bool = True,
    ) -> None:
        """The `HuggingfaceLoader` class defines a LightningDataModule for
        timeseries dataset of the GOES Xray one minute average data.

        Args:
            dataset_path (`str`):
                The path of the Huggingface dataset folder.
            test_size (`float`, *optional*, defaults to 0.2):
                Used to spit the dataset
            shuffle (`bool`, *optional*, defaults to False):
                Shuffle the dataset.
            batch_size (`int`, *optional*, defaults to 512):
                The global batch size, it will be divided by the number of GPUs.
            num_workers (`int`, *optional*, defaults to 10):
                The number of workers for the DataLoader.
            augment (`bool`, *optional*, defaults to True):
                Augment the dataset.
            density (`bool`, *optional*, defaults to True):
                Stack density field in the tensor.                
        """
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.batch_size = self.hparams.batch_size // dist.get_world_size()
        else:
            self.batch_size = self.hparams.batch_size

        self.dset = datasets.load_from_disk(self.hparams.dataset_path)

        if self.is_distributed:
            self.dset = split_dataset_by_node(
                self.dset, rank=dist.get_rank(), world_size=dist.get_world_size()
            )
        if stage == "validation":
            self.dset = self.dset.train_test_split(
                test_size=0.001,
                shuffle=False,
            )
        else:
            self.dset = self.dset.train_test_split(
                test_size=self.hparams.test_size,
                shuffle=False,
            )

        self.dset = self.dset.with_format("torch")

        self.train_dataset = AugmentedDataset(self.dset["train"], augment=self.hparams.augment, density=self.hparams.density)
        self.val_dataset = AugmentedDataset(self.dset["test"], augment=False, density=self.hparams.density)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )