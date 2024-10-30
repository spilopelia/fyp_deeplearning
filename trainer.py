import torch
import lightning.pytorch as pl
from datapile import FastPMPile
from model import Lpt2NbodyNetLightning
import yaml
import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from evaluation import SlicePlotCallback
import os
# Function to load the YAML configuration file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Argument parser to accept the config file path from the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Trainer with YAML configuration and WandB integration")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    return parser.parse_args()

# Main function to run the training
def main():
    # Parse command line arguments
    args = parse_args()
    config = load_config(args.config)
    config_file_name = os.path.basename(args.config)  # Get the file name
    config_file_name = os.path.splitext(config_file_name)[0]  # Remove the extension
    model = Lpt2NbodyNetLightning(**config['model'])

    # Extract data parameters from the config
    data_module = FastPMPile(**config['data'])

    # Extract trainer parameters from the config
    gpus = config['trainer']['gpus'] if torch.cuda.is_available() else None
    max_epochs = config['trainer']['max_epochs']
    num_nodes = config['trainer']['num_nodes']
    
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity', None),
        log_model=config['wandb'].get('log_model', False),
        save_dir=config['wandb'].get('save_dir', './wandb_logs')  # Optional save directory
    )
    # Get the current WandB run ID
    # Create a checkpoint directory using the WandB run ID
    checkpoint_dir = os.path.join('checkpoints', config_file_name)

    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_epoch_loss',
        dirpath=checkpoint_dir,
        filename='best-checkpoint-{epoch:02d}',
        save_top_k=1,
        mode='min',
        save_last=True,
        verbose=True
    )

    sliceplot_callback = SlicePlotCallback()
    
    # Initialize the PyTorch Lightning trainer
    strategy = 'ddp' if (gpus is not None and gpus > 1) else 'auto'
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=gpus,
        accelerator="gpu",
        logger=wandb_logger,
        num_nodes=num_nodes,
        strategy=strategy,
        callbacks=[checkpoint_callback, sliceplot_callback],
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Optionally test the model
    # trainer.test(datamodule=data_module)

if __name__ == "__main__":
    main()
