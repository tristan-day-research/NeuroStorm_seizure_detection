import io
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
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

        return fft_data, mask

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
        fft_result = rfft(patches, n=self.fft_size, dim=1)
        return torch.abs(fft_result)  # Return magnitude spectrum

# Visualization Function (Separate from Data Loader)
def visualize_eeg_and_fft(eeg_tensor, fft_tensor, fft_size, sample_idx=0):
    raw_signal = eeg_tensor[sample_idx, :, 0].cpu().numpy()  # [patch_size]
    fft_signal = fft_tensor[sample_idx, :, 0].cpu().numpy()  # [fft_size // 2 + 1]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(raw_signal)
    plt.title("Raw EEG Signal (Time Domain)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.plot(fft_signal)
    plt.title("FFT of EEG Signal (Frequency Domain)")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()
