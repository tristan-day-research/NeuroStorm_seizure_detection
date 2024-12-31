import io
import matplotlib.pyplot as plt
import torch
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import storage
from torch.utils.data import Dataset
from torch.fft import rfft


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
