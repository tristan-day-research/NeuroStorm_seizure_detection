import io
import gc
import matplotlib.pyplot as plt
import torch
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import storage
import torch
from torch.utils.data import Dataset
from torch.fft import rfft

import google.resumable_media.common
import logging
import time
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=0.1, highcut=75, fs=200, order=4):
    """
    Apply bandpass filter to EEG data.
    Args:
        data (torch.Tensor): EEG signal of shape (batch, channels, time).
        lowcut (float): Lower bound of the filter.
        highcut (float): Upper bound of the filter.
        fs (int): Sampling frequency (Hz).
        order (int): Filter order.
    Returns:
        torch.Tensor: Filtered EEG data.
    """
    b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    filtered_data = torch.tensor(filtfilt(b, a, data.cpu().numpy(), axis=-1), device=data.device)
    return filtered_data


# def preprocess_eeg(patches, fft_size, apply_dc_offset_removal=True,
#                    apply_window=True, window_type='hann', normalize_eeg=True, normalize_fft=False,
#                    apply_bandpass=True, fs=200, lowcut=0.1, highcut=75):
#     """
#     Preprocess EEG patches by applying DC offset removal, windowing, bandpass filtering, and FFT normalization.
#     Args:
#         patches (torch.Tensor): EEG patches (batch, channels, time).
#         fft_size (int): FFT size.
#         apply_dc_offset_removal (bool): Whether to remove DC offset.
#         apply_window (bool): Whether to apply a windowing function.
#         window_type (str): Type of window ('hann', 'hamming').
#         normalize (bool): Whether to normalize patches.
#         apply_bandpass (bool): Apply bandpass filter to EEG.
#         fs (int): Sampling frequency (Hz).
#         lowcut (float): Low cut-off frequency for bandpass.
#         highcut (float): High cut-off frequency for bandpass.
#     Returns:
#         torch.Tensor: Preprocessed FFT magnitude.
#     """


#     print("preprocessing")

#     # 1. Remove DC Offset
#     if apply_dc_offset_removal:
#         patches = patches - patches.mean(dim=-1, keepdim=True)

#     # 2. Apply Bandpass Filtering
#     if apply_bandpass:
#         patches = bandpass_filter(patches, lowcut, highcut, fs)

#     # 3. Normalize EEG Patches
#     if normalize_eeg:
#         print("normalizing EEG")
#         mean = patches.mean(dim=-1, keepdim=True)
#         std = patches.std(dim=-1, keepdim=True)
#         patches = (patches - mean) / (std + 1e-8)

#     # 4. Apply Windowing
#     if apply_window:
#         if window_type == 'hann':
#             window = torch.hann_window(patches.shape[-1], device=patches.device)
#         elif window_type == 'hamming':
#             window = torch.hamming_window(patches.shape[-1], device=patches.device)
#         else:
#             raise ValueError(f"Unsupported window type: {window_type}")
#         patches = patches * window

#     # 5. Perform FFT
#     fft_result = torch.fft.rfft(patches, n=fft_size, dim=-1)
#     fft_magnitude = torch.abs(fft_result)

#     # 6. Normalize FFT Magnitude
#     if normalize_fft:
#         fft_magnitude = fft_magnitude / fft_size
#         mean = fft_magnitude.mean(dim=(0, 1), keepdim=True)
#         std = fft_magnitude.std(dim=(0, 1), keepdim=True)
#         fft_magnitude = (fft_magnitude - mean) / (std + 1e-8)

#     return fft_magnitude

def preprocess_eeg(patches, fft_size, apply_dc_offset_removal=True,
                   apply_window=True, window_type='hann', normalize_eeg=True, normalization_type='zscore',
                   normalize_fft=False, apply_bandpass=True, fs=200, lowcut=0.1, highcut=75):
    """
    Preprocess EEG patches by applying DC offset removal, windowing, bandpass filtering, and FFT normalization.
    Args:
        patches (torch.Tensor): EEG patches (batch, channels, time).
        fft_size (int): FFT size.
        apply_dc_offset_removal (bool): Whether to remove DC offset.
        apply_window (bool): Whether to apply a windowing function.
        window_type (str): Type of window ('hann', 'hamming').
        normalize_eeg (bool): Whether to normalize EEG patches.
        normalization_type (str): 'zscore' or 'minmax' for EEG normalization.
        apply_bandpass (bool): Apply bandpass filter to EEG.
        fs (int): Sampling frequency (Hz).
        lowcut (float): Low cut-off frequency for bandpass.
        highcut (float): High cut-off frequency for bandpass.
    Returns:
        torch.Tensor: Preprocessed FFT magnitude and normalized EEG patches.
    """

    print("preprocessing")

    # 1. Remove DC Offset
    if apply_dc_offset_removal:
        patches = patches - patches.mean(dim=-1, keepdim=True)

    # 2. Apply Bandpass Filtering
    if apply_bandpass:
        patches = bandpass_filter(patches, lowcut, highcut, fs)

    # 3. Normalize EEG Patches
    if normalize_eeg:
        print("normalizing EEG")
        if normalization_type == 'zscore':
            mean = patches.mean(dim=-1, keepdim=True)
            std = patches.std(dim=-1, keepdim=True)
            patches = (patches - mean) / (std + 1e-8)
        elif normalization_type == 'minmax':
            min_val = patches.min(dim=-1, keepdim=True).values
            max_val = patches.max(dim=-1, keepdim=True).values
            patches = 2 * ((patches - min_val) / (max_val - min_val + 1e-8)) - 1  # Scale to [-1, 1]
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

    # 4. Apply Windowing
    if apply_window:
        if window_type == 'hann':
            window = torch.hann_window(patches.shape[-1], device=patches.device)
        elif window_type == 'hamming':
            window = torch.hamming_window(patches.shape[-1], device=patches.device)
        else:
            raise ValueError(f"Unsupported window type: {window_type}")
        patches = patches * window

    # 5. Perform FFT
    fft_result = torch.fft.rfft(patches, n=fft_size, dim=-1)
    fft_magnitude = torch.abs(fft_result)

    # 6. Normalize FFT Magnitude
    if normalize_fft:
        fft_magnitude = fft_magnitude / fft_size
        mean = fft_magnitude.mean(dim=(0, 1), keepdim=True)
        std = fft_magnitude.std(dim=(0, 1), keepdim=True)
        fft_magnitude = (fft_magnitude - mean) / (std + 1e-8)

    return fft_magnitude, patches  # Return both FFT and normalized EEG



# def normalize_fft(fft_data, method='zscore', epsilon=1e-8):
#     """
#     Normalize FFT data.

#     Args:
#         fft_data (torch.Tensor): FFT data of shape (batch, time, channels).
#         method (str): Normalization method ('zscore', 'minmax', 'log').
#         epsilon (float): Small value to avoid division by zero.

#     Returns:
#         torch.Tensor: Normalized FFT data.
#     """
#     if method == 'zscore':
#         # Z-score normalization (mean 0, std 1)
#         mean = fft_data.mean(dim=(0, 1), keepdim=True)
#         std = fft_data.std(dim=(0, 1), keepdim=True)
#         normalized_fft = (fft_data - mean) / (std + epsilon)

#     elif method == 'minmax':
#         # Min-Max scaling to [0, 1]
#         min_val = fft_data.min(dim=(0, 1), keepdim=True).values
#         max_val = fft_data.max(dim=(0, 1), keepdim=True).values
#         normalized_fft = (fft_data - min_val) / (max_val - min_val + epsilon)

#     elif method == 'log':
#         # Logarithmic scaling (compress large values)
#         normalized_fft = torch.log1p(fft_data)

#         # Optional Z-score after log
#         mean = normalized_fft.mean(dim=(0, 1), keepdim=True)
#         std = normalized_fft.std(dim=(0, 1), keepdim=True)
#         normalized_fft = (normalized_fft - mean) / (std + epsilon)

#     else:
#         raise ValueError(f"Unsupported normalization method: {method}")

#     return normalized_fft


def pad_tensor_with_nan_check(tensor, max_length, max_invalid_ratio=0.3):
    invalid_mask = torch.isnan(tensor) | torch.isinf(tensor)
    invalid_ratio = invalid_mask.sum().item() / tensor.numel()

    # Mask for valid entries
    valid_mask = ~invalid_mask

    if invalid_ratio > max_invalid_ratio:
        print(f"[INFO] Zero-padding patch with {invalid_ratio:.2%} NaN/Inf")
        tensor[invalid_mask] = 0  # Zero out NaNs/Infs instead of skipping
    else:
        tensor[invalid_mask] = 0  # Regular zero padding

    pad_size = max_length - tensor.shape[0]
    if pad_size > 0:
        # Zero pad the tensor to ensure uniform shape
        padding = torch.zeros((pad_size, *tensor.shape[1:]), device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
        
        # Pad the mask to track the valid regions
        mask_padding = torch.zeros((pad_size, *valid_mask.shape[1:]), device=tensor.device, dtype=torch.bool)
        valid_mask = torch.cat((valid_mask, mask_padding), dim=0)

    return tensor, valid_mask


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
        
    # Unpack the batch correctly - each item is already a tuple
    eeg_patches, fft_data, mask = [], [], []

    
    for item in batch:
        e, f, m = item

        eeg_patches.append(e)
        fft_data.append(f)
        mask.append(m)


    # for item in batch:
    #     e, f, m = item
    #     eeg_patches.append(e)
    #     fft_data.append(f)
    #     mask.append(m)

    max_patches = max(tensor.shape[0] for tensor in eeg_patches)
    
    # Rest of your padding and stacking logic
    eeg_results = [pad_tensor_with_nan_check(t, max_patches) for t in eeg_patches]
    fft_results = [pad_tensor_with_nan_check(t, max_patches) for t in fft_data]
    mask_results = [pad_tensor_with_nan_check(t, max_patches) for t in mask]
    
    eeg_patches, eeg_mask = zip(*eeg_results)
    fft_data, fft_mask = zip(*fft_results)
    mask, mask_pad = zip(*mask_results)
    
    return (torch.stack(eeg_patches), 
            torch.stack(fft_data),
            torch.stack(mask),
            torch.stack(eeg_mask))


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

    # def __getitem__(self, idx):
    #     # Load parquet file from GCS
    #     blob = self.files[idx]
    #     byte_stream = io.BytesIO()
    #     blob.download_to_file(byte_stream)
    #     byte_stream.seek(0)

    #     # Read parquet file
    #     eeg_data = pq.read_table(byte_stream).to_pandas()
    #     eeg_tensor = torch.tensor(eeg_data.values, dtype=torch.float32)  # Shape: (time, channels)

    #     # Segment EEG into patches
    #     patches, mask = self._segment_into_patches(eeg_tensor)
        
    #     # Perform FFT on each patch
    #     fft_data = self._apply_fft(patches)

    #     # Return both raw EEG patches and FFT data
    #     return patches, fft_data, mask

    # def __getitem__(self, idx):
    #     # Load parquet file from GCS
    #     blob = self.files[idx]
    #     byte_stream = io.BytesIO()
    #     blob.download_to_file(byte_stream)
    #     byte_stream.seek(0)

    #     # Read parquet file
    #     eeg_data = pq.read_table(byte_stream).to_pandas()
    #     eeg_tensor = torch.tensor(eeg_data.values, dtype=torch.float32)  # Shape: (time, channels)

    #     # Segment EEG into patches
    #     patches, mask = self._segment_into_patches(eeg_tensor)
        
    #     # Apply Preprocessing (DC offset removal, windowing, normalization)
    #     fft_data = preprocess_eeg(
    #         patches=patches,
    #         fft_size=self.fft_size,  # Use the fft_size from the dataset instance
    #         apply_dc_offset_removal=True,
    #         apply_window=True,
    #         window_type='hann',
    #         normalize=True
    #     )

    #     # Return preprocessed FFT data along with the raw patches and mask
    #     return patches, fft_data, mask



    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Load parquet file from GCS
                blob = self.files[idx]
                byte_stream = io.BytesIO()
                blob.download_to_file(byte_stream)
                byte_stream.seek(0)

                # Read parquet file
                eeg_data = pq.read_table(byte_stream).to_pandas()
                eeg_tensor = torch.tensor(eeg_data.values, dtype=torch.float32)  # Shape: (time, channels)

                # Apply Bandpass Filter to Full EEG Signal (Before Segmenting)
                # eeg_tensor = bandpass_filter(eeg_tensor, lowcut=0.1, highcut=75, fs=200)

                # Segment EEG into patches
                patches, mask = self._segment_into_patches(eeg_tensor)
                
                # # Apply Preprocessing
                # fft_data = preprocess_eeg(
                #     patches=patches,
                #     fft_size=self.fft_size,
                #     apply_dc_offset_removal=False,
                #     apply_window=False,
                #     window_type='hann',
                #     normalize=False
                # )

                # Apply Preprocessing (Normalization + Bandpass + FFT)
                fft_data, normalized_patches = preprocess_eeg(
                    patches=patches,
                    fft_size=self.fft_size,
                    apply_dc_offset_removal=False,
                    apply_window=False,
                    window_type='hann',
                    normalize_eeg=True,
                    normalization_type='minmax',  # Use min-max to scale to [-1, 1]
                    normalize_fft=False,
                    apply_bandpass=False,
                    fs=200,
                    lowcut=0.1,
                    highcut=75
                )
                # Return preprocessed FFT data along with the raw patches and mask
                return normalized_patches, fft_data, mask
            
            except google.resumable_media.common.DataCorruption as e:
                logging.warning(f"[Warning] Data corruption detected in file: {self.files[idx].name}. Attempt {attempt + 1}")
                time.sleep(2 ** attempt)  # Exponential back-off

            except Exception as e:
                logging.error(f"[Error] Failed to process file: {self.files[idx].name}. Error: {e}")
                return None

        # If all retries fail, skip the sample
        logging.error(f"[Error] Skipping corrupted file after {max_retries} attempts: {self.files[idx].name}")
        return None

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
