


def visualize_fft(target_amp, target_phase, output_amp, output_phase, frequency_range=None):
    """
    Visualizes the amplitude and phase spectrum of an EEG patch for both target and output.
    """

    target_amp = target_amp.detach().cpu().numpy()

    target_phase = target_phase.detach().cpu().numpy()

    output_amp = output_amp.detach().cpu().numpy()

    output_phase = output_phase.detach().cpu().numpy()


    frequencies = np.linspace(0, 0.5, target_amp.shape[-1])  # Assuming sampling rate of 1 for simplicity

    if frequency_range is not None:
        freq_indices = np.where((frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1]))
        frequencies = frequencies[freq_indices]
        target_amp, target_phase = target_amp[freq_indices], target_phase[freq_indices]
        output_amp, output_phase = output_amp[freq_indices], output_phase[freq_indices]

    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, target_amp[1], label='Target Amplitude')
    plt.plot(frequencies, output_amp[1], label='Output Amplitude', linestyle='--')
    plt.title('Amplitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(frequencies, target_phase[1], label='Target Phase')
    plt.plot(frequencies, output_phase[1], label='Output Phase', linestyle='--')
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Radians)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    del target_amp, target_phase, output_amp, output_phase




import torch
import gcsfs
from datetime import datetime



# ______This is what I used before, a version that doesn't store the model architechture. _____

# def save_model(model, optimizer, epoch, batch, model_path='your_bucket/folder', string_to_add='sample_text'):
#     # Timestamp or other unique identifiers could be useful here
#     timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#     filename = f"{model_path}/{string_to_add}_model_epoch_{epoch}_{timestamp}.pth"

#     # Prepare the state dict
#     state = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }

#     with fs.open(filename, 'wb') as f:
#         torch.save(state, f)

#     print(f"Model saved to {filename}")

# ______ This version does save the model architechture ______________________
def save_model(model, optimizer, epoch, batch, model_path='your_bucket/folder', string_to_add='sample_text'):
    fs = gcsfs.GCSFileSystem(project='your_project_id')
    # Timestamp or other unique identifiers could be useful here
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    filename = f"{model_path}/{string_to_add}_model_epoch_{epoch}_{timestamp}.pth"

    # Prepare the state dict
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_architecture': model,  # Save the model architecture
    }

    with fs.open(filename, 'wb') as f:
        torch.save(state, f)

    print(f"Model saved to {filename}")



# string_to_add = "test_model"
# model_folder = 'labram_models'
# model_path = f"{BUCKET_NAME}/{model_folder}"
# save_model(model, optimizer, epoch, batch, path=model_path, string_to_add=string_to_add)


def plot_per_batch(loss_type, losses):
        window_size = 10  # Size of the moving average window
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

        # Plot the original losses
        plt.figure(figsize=(6, 3))
        plt.plot(losses, label='Training Loss')

        # Plot the moving average
        x_data = np.arange(len(moving_avg))  # Use the length of moving_avg as the x-axis range
        plt.plot(x_data, moving_avg, label=f'{window_size}-Batch Moving Average', color='red')

        # plt.plot(np.arange(window_size-1, len(losses)), moving_avg, label=f'{window_size}-Batch Moving Average', color='red')

        # plt.xlabel('Batch Number')
        # plt.ylabel('Loss')
        plt.title(f'{loss_type}')
        plt.legend()

        # Limit y-axis from 0 to 1000
        plt.ylim(0, 1000)
        plt.show()






def load_model(model, optimizer, model_load_path):
    with fs.open(model_load_path, 'rb') as f:
        checkpoint = torch.load(f)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


model_to_load = 'model_epoch_0_2024-04-06_00-46-33.pth'
model_folder = 'labram_models'
model_load_path =  f"{BUCKET_NAME}/{model_folder}/{model_to_load}"

model, optimizer, epoch = load_model(model, optimizer, model_load_path)

