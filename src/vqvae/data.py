
from einops import rearrange
from torch.nn import Conv1d
from torch.utils.data import Dataset
import io
import h5py
import time
import google.cloud.storage
import torch.nn as nn


def contains_invalid_values(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def check_tensor_for_nan_inf(tensor, tensor_name="tensor"):
    # Check for NaN and infinity
    if torch.isnan(tensor).any():
        print(f"{tensor_name} contains NaN .")
    if torch.isinf(tensor).any():
        print(f"{tensor_name} contains  Infinity.")


class EEGSegmenter(nn.Module):
    def __init__(self, patch_size=200, stride=None, n_channels=19):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.n_channels = n_channels
        # Conv1d to segment EEG data into patches
        self.patcher = Conv1d(in_channels=1, out_channels=n_channels, kernel_size=self.patch_size, stride=self.stride)

    def forward(self, x):
        # Rearrange for Conv1d and segment
        x = rearrange(x, 'b c t -> (b c) 1 t')
        x = self.patcher(x)
        # Rearrange back to (batch, channels, patches, emb_dim)
        x = rearrange(x, '(b c) e p -> b c p e', c=self.n_channels)
        return x


def custom_collate_fn(batch):
    return batch  # Directly returning the list of tensors

class EEGDataset(Dataset):
    def __init__(self, bucket_name, blob_prefix, transform=None):
        """
        Args:
            bucket_name (string): Name of the GCS bucket.
            blob_prefix (string): Prefix of the blobs (files) to read.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.bucket_name = bucket_name
        self.blob_prefix = blob_prefix
        self.transform = transform
        self.storage_client = google.cloud.storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.blobs = list(self.bucket.list_blobs(prefix=blob_prefix))
        self.file_names = [blob.name for blob in self.blobs]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                blob = self.blobs[idx]
                byte_stream = io.BytesIO()
                file_name = blob.name  # Assuming the blob object has the 'name' attribute that contains the file name
                blob.download_to_file(byte_stream)
                byte_stream.seek(0)

                try:
                    with h5py.File(byte_stream, 'r') as f:
                        eeg_data = f['processed_eeg_data'][:]
                        eeg_data = torch.tensor(eeg_data, dtype=torch.float)

                    # Add a batch dimension to eeg_data
                    check_tensor_for_nan_inf(eeg_data, "eeg_data after opened from .h5 file")
                    eeg_data = eeg_data.unsqueeze(0)

                    if self.transform:
                        eeg_tensor, mask = self.transform(eeg_data)
                        return file_name, eeg_tensor.squeeze(0).detach(), mask.squeeze(0).detach() # Detach gradients

                    eeg_tensor, mask = eeg_data[0].detach(), eeg_data[1].detach()

                    print(type(eeg_tensor), type(mask))
                    print(eeg_tensor.shape, mask.shape)

                    # Optionally, remove the batch dimension here if your model expects it
                    eeg_tensor = eeg_tensor.squeeze(0)

                    check_tensor_for_nan_inf(eeg_tensor, tensor_name="eeg_tensor IN DATA LOADER get_item")
                    check_tensor_for_nan_inf(mask, tensor_name="mask IN DATA LOADER get_item")

                    if contains_invalid_values(eeg_tensor) or contains_invalid_values(mask):
                        print(f"Invalid values detected in file: {self.file_names[idx]}")

                    return file_name, eeg_tensor.detach(), mask.detach() # Detach gradients


                except Exception as e:
                    logging.error(f"Skipping index {idx}, file {file_path} due to error: {e}")
                    return None  # Or some default value/placeholder

            except google.resumable_media.common.DataCorruption as e:
                print(f"Attempt {attempt+1} failed with DataCorruption: {e}")
                time.sleep(2**attempt)  # Exponential back-off
            except Exception as e:
                print("RuntimeError raised")
                raise RuntimeError(f"Skipping index {idx} due to data loading error: {e}")
        raise RuntimeError(f"Failed to download after {max_retries} attempts, skipping index {idx}")


class ToPatches(nn.Module):
    def __init__(self, patch_size=200, stride=150, max_patches=1000):
        super(ToPatches, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.max_patches = max_patches

    def forward(self, x):
        batch_size, num_channels, time_length = x.shape

        # Define the target number of patches (here fixed to `max_patches`)
        target_patches = self.max_patches

        # Initialize tensors to hold the batched patches and masks
        batch_patches = torch.zeros((batch_size, num_channels, target_patches, self.patch_size),
                                    dtype=x.dtype, device=x.device)
        batch_masks = torch.zeros((batch_size, num_channels, target_patches), dtype=torch.float32, device=x.device)

        for b in range(batch_size):
            # Compute the actual number of patches that can be extracted from the current EEG sequence
            actual_num_patches = min((time_length - self.patch_size) // self.stride + 1, target_patches)

            for i in range(actual_num_patches):
                start = i * self.stride
                end = start + self.patch_size
                batch_patches[b, :, i, :] = x[b, :, start:end]

            # Update the mask to indicate valid patches
            batch_masks[b, :, :actual_num_patches] = 1

        return batch_patches, batch_masks

