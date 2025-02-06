import lightning.pytorch as pl
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.gridspec import GridSpec


class SlicePlotCallback(pl.Callback):
    def look_dis_slice(self, trainer, net1, net2, net3, batch_idx, axis, slice_index):
        # Turn off axis for the overall figure
        plt.axis('off')
        matplotlib.rc('font', size=12)

        # Ensure that the last dimension of the input arrays is 3
        net1 = net1[0,:,:,:,:].detach().cpu().numpy()
        net2 = net2[0,:,:,:,:].detach().cpu().numpy()
        net3 = net3[0,:,:,:,:].detach().cpu().numpy()
        direction=['x','y','z']

        # Create the figure and define the grid layout
        fig = plt.figure(figsize=(20, 10))  # Adjusted the figure size for better visualization
        gs = GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.1])

        # Calculate the amplitude limits for the color scaling
        amp_low = min(np.percentile(net1[axis, :, :, slice_index], 5),
                    np.percentile(net2[axis, :, :, slice_index], 5),
                    np.percentile(net3[axis, :, :, slice_index], 5))
        amp_high = max(np.percentile(net1[axis, :, :, slice_index], 95),
                    np.percentile(net2[axis, :, :, slice_index], 95),
                    np.percentile(net3[axis, :, :, slice_index], 95))

        # Calculate the amplitude limits for the residuals
        amp_low1 = min(np.percentile((net1[axis, :, :, slice_index] - net2[axis, :, :, slice_index]), 5),
                    np.percentile((net1[axis, :, :, slice_index] - net3[axis, :, :, slice_index]), 5))
        amp_high1 = max(np.percentile((net1[axis, :, :, slice_index] - net2[axis, :, :, slice_index]), 95),
                        np.percentile((net1[axis, :, :, slice_index] - net3[axis, :, :, slice_index]), 95))

        # Set a common amplitude for color scaling
        amp = min(np.abs(amp_low), amp_high)

        # Plot the first row (FastPM, ZA, U-Net)
        plt.subplot(gs[0, 0])
        im1 = plt.imshow(net1[axis, :, :, slice_index], cmap="coolwarm", vmin=-amp, vmax=amp)
        plt.axis('off')
        plt.title(f'FastPM (direction={direction[axis]}')

        plt.subplot(gs[0, 1])
        plt.imshow(net2[axis, :, :, slice_index], cmap="coolwarm", vmin=-amp, vmax=amp)
        plt.axis('off')
        plt.title(f'ZA (direction={direction[axis]}')

        plt.subplot(gs[0, 2])
        plt.imshow(net3[axis, :, :, slice_index], cmap="coolwarm", vmin=-amp, vmax=amp)
        plt.axis('off')
        plt.title(f'U-Net (direction={direction[axis]}')

        # Add a colorbar for the first row
        cbax1 = plt.subplot(gs[0, 3])
        cbar1 = fig.colorbar(mappable=im1, cax=cbax1, orientation='vertical', ticklocation='right')
        cbar1.ax.tick_params(labelsize=12)

        # Plot the second row (Residuals: FastPM - ZA, FastPM - U-Net)
        plt.subplot(gs[1, 1])
        im2 = plt.imshow(net1[axis, :, :, slice_index] - net2[axis, :, :, slice_index], cmap="coolwarm", vmin=amp_low1, vmax=amp_high1)
        plt.axis('off')
        plt.title(r'FastPM $-$ ZA')

        plt.subplot(gs[1, 2])
        plt.imshow(net1[axis, :, :, slice_index] - net3[axis, :, :, slice_index], cmap="coolwarm", vmin=amp_low1, vmax=amp_high1)
        plt.axis('off')
        plt.title(r'FastPM $-$ U-Net')

        # Add a colorbar for the second row (residuals)
        cbax2 = plt.subplot(gs[1, 3])
        cbar2 = fig.colorbar(mappable=im2, cax=cbax2, orientation='vertical', ticklocation='right')
        cbar2.ax.tick_params(labelsize=12)

        # Adjust layout to avoid overlap
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        # Log the figure to the trainer's logger (e.g., WandB or TensorBoard)
        log_key = f"batch_{batch_idx}_axis_{direction[axis]}_slice_{slice_index}"
        trainer.logger.experiment.log({log_key: fig}, step=trainer.global_step)

        # Close the figure to avoid memory leaks
        plt.close(fig)

    def plot_slices_and_residuals(self, trainer, input_tensor, target_tensor, prediction_tensor, batch_idx, axis, slice_index):
            """
            Plots slices of the input, target, prediction, and residual (difference) for visualization.
            
            Args:
            - trainer: The PyTorch Lightning trainer.
            - pl_module: The LightningModule (your model).
            - input_tensor: The 5D input tensor (batch_size, channel, 32, 32, 32).
            - target_tensor: The 5D target tensor (batch_size, channel, 32, 32, 32).
            - prediction_tensor: The 5D predicted tensor (batch_size, channel, 32, 32, 32).
            - batch_idx: The current batch index (used for logging).
            """
            # Convert tensors to numpy arrays (assuming batch size of 1 for simplicity)
            input_array = input_tensor[0,:,:,:,:].detach().cpu().numpy()
            target_array = target_tensor[0,:,:,:,:].detach().cpu().numpy()
            prediction_array = prediction_tensor[0,:,:,:,:].detach().cpu().numpy()
            direction=['x','y','z']

            # Plot slices
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # Select the slice along the specified axis
            input_slice = input_array[axis,:,:,slice_index]
            target_slice = target_array[axis,:,:,slice_index]
            prediction_slice = prediction_array[axis,:,:,slice_index]
            residual_slice = target_slice - prediction_slice

            # Plot input, target, prediction, and residual slices
            im0 = axes[0].imshow(input_slice, cmap='coolwarm')
            axes[0].set_title(f'ZA Slice (direction={direction[axis]}, z_index={slice_index})')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)  # Add colorbar to the first plot

            im1 = axes[1].imshow(target_slice, cmap='coolwarm')
            axes[1].set_title(f'FastPM Slice (direction={direction[axis]}, z_index={slice_index})')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar to the second plot

            im2 = axes[2].imshow(prediction_slice, cmap='coolwarm')
            axes[2].set_title(f'UNet Slice (direction={direction[axis]}, z_index={slice_index})')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)  # Add colorbar to the third plot

            im3 = axes[3].imshow(residual_slice, cmap='coolwarm')
            axes[3].set_title(f'Residual (Difference) Slice')
            plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)  # Add colorbar to the fourth plot

            # Save the plot as an image and log it to WandB (or another logger)
            plt.tight_layout()

            log_key = f"slice_residuals_batch_{batch_idx}_axis_{direction[axis]}_slice_{slice_index}"
            trainer.logger.experiment.log({log_key: fig}, step=trainer.global_step)

            # Close the figure to avoid memory leaks
            plt.close(fig)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        This method is called at the end of every validation batch.
        
        Args:
        - trainer: The PyTorch Lightning trainer.
        - pl_module: The LightningModule (your model).
        - outputs: The outputs from the validation step.
        - batch: The current batch (input, target).
        - batch_idx: The index of the current batch.
        - dataloader_idx: The index of the dataloader if multiple dataloaders are used.
        """
        input_tensor, target_tensor = batch
        prediction_tensor = outputs  # Assuming the output of validation_step is the prediction

        if batch_idx == 0:
            #self.plot_slices_and_residuals(trainer, input_tensor, target_tensor, prediction_tensor, batch_idx, axis=0, slice_index=16)
            #self.plot_slices_and_residuals(trainer, input_tensor, target_tensor, prediction_tensor, batch_idx, axis=1, slice_index=16)
            self.plot_slices_and_residuals(trainer, input_tensor, target_tensor, prediction_tensor, batch_idx, axis=2, slice_index=16)
            #self.look_dis_slice(trainer, net1=target_tensor, net2=input_tensor, net3=prediction_tensor, batch_idx=batch_idx, axis=0, slice_index=16)
            #self.look_dis_slice(trainer, net1=target_tensor, net2=input_tensor, net3=prediction_tensor, batch_idx=batch_idx, axis=1, slice_index=16)
            self.look_dis_slice(trainer, net1=target_tensor, net2=input_tensor, net3=prediction_tensor, batch_idx=batch_idx, axis=2, slice_index=16)
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        This method is called at the end of every validation batch.
        
        Args:
        - trainer: The PyTorch Lightning trainer.
        - pl_module: The LightningModule (your model).
        - outputs: The outputs from the validation step.
        - batch: The current batch (input, target).
        - batch_idx: The index of the current batch.
        - dataloader_idx: The index of the dataloader if multiple dataloaders are used.
        """
        input_tensor, target_tensor = batch
        prediction_tensor = outputs  # Assuming the output of validation_step is the prediction

        if batch_idx == 0:
            #self.plot_slices_and_residuals(trainer, input_tensor, target_tensor, prediction_tensor, batch_idx, axis=0, slice_index=16)
            #self.plot_slices_and_residuals(trainer, input_tensor, target_tensor, prediction_tensor, batch_idx, axis=1, slice_index=16)
            self.plot_slices_and_residuals(trainer, input_tensor, target_tensor, prediction_tensor, batch_idx, axis=2, slice_index=16)
            #self.look_dis_slice(trainer, net1=target_tensor, net2=input_tensor, net3=prediction_tensor, batch_idx=batch_idx, axis=0, slice_index=16)
            #self.look_dis_slice(trainer, net1=target_tensor, net2=input_tensor, net3=prediction_tensor, batch_idx=batch_idx, axis=1, slice_index=16)
            self.look_dis_slice(trainer, net1=target_tensor, net2=input_tensor, net3=prediction_tensor, batch_idx=batch_idx, axis=2, slice_index=16)