import torch
import lightning.pytorch as pl
from datapile import FastPMPile, HuggingfaceLoader
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
    
    # Adding new arguments
    parser.add_argument('--gpus', type=int, help='Specify the GPU number to use (default: 0)')
    parser.add_argument('--num_nodes', type=int, help='Specify the GPU number to use (default: 0)')
    parser.add_argument('--num_workers', type=int, help='Specify the number of workers (default: 1)')
    
    return parser.parse_args()

# Main function to run the training
def main():

    # Parse command line arguments
    args = parse_args()
    config = load_config(args.config)

    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers  # Update the workers count in the config

    if args.gpus is not None:
        config['trainer']['gpus'] = args.gpus  # Update the GPU count in the config

    if args.num_nodes is not None:
        config['trainer']['num_nodes'] = args.num_nodes  # Update the number of nodes in the config

    config_file_name = os.path.basename(args.config)  # Get the file name
    config_file_name = os.path.splitext(config_file_name)[0]  # Remove the extension
    model = Lpt2NbodyNetLightning(**config['model'])

    # Extract data parameters from the config
    # data_module = FastPMPile(**config['data'])
    data_module = HuggingfaceLoader(**config['data']) # faster data pile
    # Extract trainer parameters from the config
    gpus = config['trainer']['gpus'] if torch.cuda.is_available() else None
    max_epochs = config['trainer']['max_epochs']
    num_nodes = config['trainer']['num_nodes']
    ckpt_path = config['trainer'].get('ckpt_path', None)
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity', None),
        log_model=config['wandb'].get('log_model', False),
        save_dir=config['wandb'].get('save_dir', './wandb_logs'),  # Optional save directory
        id=config['wandb'].get('id', None),
        resume=config['wandb'].get('resume', None),
    )
    # Get the current WandB run ID
    # Create a checkpoint directory using the WandB run ID
    checkpoint_dir = os.path.join('new_checkpoints', config_file_name)

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
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

    # Optionally test the model
    # trainer.test(datamodule=data_module)

if __name__ == "__main__":
    main()
