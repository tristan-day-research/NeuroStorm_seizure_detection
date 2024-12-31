import io
import matplotlib.pyplot as plt
import torch
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import storage
from torch.utils.data import Dataset
from torch.fft import rfft

def normalize_fft(fft_data, method='zscore', epsilon=1e-8):
    """
    Normalize FFT data.

    Args:
        fft_data (torch.Tensor): FFT data of shape (batch, time, channels).
        method (str): Normalization method ('zscore', 'minmax', 'log').
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Normalized FFT data.
    """
    if method == 'zscore':
        # Z-score normalization (mean 0, std 1)
        mean = fft_data.mean(dim=(0, 1), keepdim=True)
        std = fft_data.std(dim=(0, 1), keepdim=True)
        normalized_fft = (fft_data - mean) / (std + epsilon)

    elif method == 'minmax':
        # Min-Max scaling to [0, 1]
        min_val = fft_data.min(dim=(0, 1), keepdim=True).values
        max_val = fft_data.max(dim=(0, 1), keepdim=True).values
        normalized_fft = (fft_data - min_val) / (max_val - min_val + epsilon)

    elif method == 'log':
        # Logarithmic scaling (compress large values)
        normalized_fft = torch.log1p(fft_data)

        # Optional Z-score after log
        mean = normalized_fft.mean(dim=(0, 1), keepdim=True)
        std = normalized_fft.std(dim=(0, 1), keepdim=True)
        normalized_fft = (normalized_fft - mean) / (std + epsilon)

    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return normalized_fft


def pad_tensor_with_nan_check(tensor, max_length, max_invalid_ratio=0.3):
    # Calculate invalid ratio (NaN or Inf)
    invalid_mask = torch.isnan(tensor) | torch.isinf(tensor)
    invalid_ratio = invalid_mask.sum().item() / tensor.numel()

    # Skip the patch if invalid ratio exceeds threshold
    if invalid_ratio > max_invalid_ratio:
        print(f"[INFO] Skipping patch - {invalid_ratio:.2%} NaN/Inf")
        return None  # Mark for removal

    # Create mask for current tensor size (before padding)
    valid_mask = ~invalid_mask

    # Pad the tensor to max length if needed
    pad_size = max_length - tensor.shape[0]
    if pad_size > 0:
        # Pad the tensor
        padding = torch.zeros((pad_size, *tensor.shape[1:]), device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
        
        # Pad the mask (expanding along the first dimension)
        mask_padding = torch.zeros((pad_size, *valid_mask.shape[1:]), device=tensor.device, dtype=torch.bool)
        valid_mask = torch.cat((valid_mask, mask_padding), dim=0)

    # Apply the mask (replace NaN/Inf values)
    tensor[~valid_mask] = 0

    return tensor


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        print("[WARNING] Entire batch skipped due to invalid patches.")
        return None  # Skip entire batch if all patches are invalid

    eeg_patches, fft_data, mask = zip(*batch)

    # Pad tensors to the same length
    max_patches = max(tensor.shape[0] for tensor in eeg_patches)

    eeg_patches = [pad_tensor_with_nan_check(t, max_patches) for t in eeg_patches]
    fft_data = [pad_tensor_with_nan_check(t, max_patches) for t in fft_data]
    mask = [pad_tensor_with_nan_check(t, max_patches) for t in mask]

    # Remove any None values after padding
    valid_entries = [(p, f, m) for p, f, m in zip(eeg_patches, fft_data, mask) if p is not None]

    if len(valid_entries) == 0:
        print("[WARNING] Batch fully invalid after padding. Skipping...")
        return None

    eeg_patches, fft_data, mask = zip(*valid_entries)

    eeg_patches = torch.stack(eeg_patches)
    fft_data = torch.stack(fft_data)
    mask = torch.stack(mask)

    return eeg_patches, fft_data, mask


class EEGDataset(Dataset):
    def __init__(self, bucket_name, gcp_file_path, patch_size, overlap, fft_size, stride=None):
        self.bucket_name = bucket_name
        self.gcp_file_path = gcp_file_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.fft_size = fft_size
        self.stride = stride or patch_size // 2

        # Initialize GCS client and get file list
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.files = list(self.bucket.list_blobs(prefix=gcp_file_path))
        self.file_names = [blob.name for blob in self.files]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Load parquet file from GCS
        blob = self.files[idx]
        byte_stream = io.BytesIO()
        blob.download_to_file(byte_stream)
        byte_stream.seek(0)

        # Read parquet file
        eeg_data = pq.read_table(byte_stream).to_pandas()
        eeg_tensor = torch.tensor(eeg_data.values, dtype=torch.float32)  # Shape: (time, channels)

        # Segment EEG into patches
        patches, mask = self._segment_into_patches(eeg_tensor)
        
        # Perform FFT on each patch
        fft_data = self._apply_fft(patches)

        # Return both raw EEG patches and FFT data
        return patches, fft_data, mask

    def _segment_into_patches(self, eeg_tensor):
        t, c = eeg_tensor.shape
        patch_list = []
        mask_list = []

        stride = self.patch_size - self.overlap
        for start in range(0, t - self.patch_size + 1, stride):
            patch = eeg_tensor[start:start + self.patch_size]
            patch_list.append(patch)
            mask_list.append(1)

        # Zero-pad if fewer patches than expected
        max_patches = (t - self.patch_size) // stride + 1
        batch_patches = torch.zeros((max_patches, self.patch_size, c), dtype=eeg_tensor.dtype)
        batch_masks = torch.zeros((max_patches,), dtype=torch.float32)

        for i, patch in enumerate(patch_list):
            batch_patches[i] = patch
            batch_masks[i] = 1

        return batch_patches, batch_masks

    def _apply_fft(self, patches):
        # Apply FFT along the time dimension (dim=1)
        fft_result = rfft(patches, n=self.fft_size, dim=1)
        return torch.abs(fft_result)  # Return magnitude spectrum



# Visualize Raw EEG (Time Domain)
def visualize_eeg(eeg_tensor, sample_idx=0, channel=0):
    raw_signal = eeg_tensor[sample_idx, :, channel].cpu().numpy()  # Select channel
    plt.figure(figsize=(10, 4))
    plt.plot(raw_signal)
    plt.title(f"EEG Signal (Channel {channel}) - Time Domain")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


# Visualize FFT (Frequency Domain)
def visualize_fft(fft_tensor, fft_size, sample_idx=0, channel=0):
    fft_signal = fft_tensor[sample_idx, :, channel].cpu().numpy()
    freqs = torch.linspace(0, fft_size // 2, fft_signal.shape[0])  # Frequency bins
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_signal)
    plt.title(f"FFT of EEG Signal (Channel {channel}) - Frequency Domain")
    plt.xlabel("Frequency (Hz or Bins)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()
