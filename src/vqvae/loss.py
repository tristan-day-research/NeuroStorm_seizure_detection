
def pad_to_power_of_two(tensor):
    # Calculate the next power of two
    original_len = tensor.size(-1)
    target_len = 2**math.ceil(math.log2(original_len))
    pad_size = target_len - original_len
    # Pad the last dimension to the next power of two
    padded_tensor = F.pad(tensor, (0, pad_size), mode='constant', value=0)
    return padded_tensor



def calculate_phase_weight(current_batch, start_batch, end_batch):
    if current_batch < start_batch:
        return 0.0  # Phase not included
    elif current_batch > end_batch:
        return 1.0  # Phase fully included
    else:
        return (current_batch - start_batch) / (end_batch - start_batch)



def z_score_normalize(data):
    mean = torch.mean(data, dim=(-2, -1), keepdim=True)
    std = torch.std(data, dim=(-2, -1), keepdim=True)
    return (data - mean) / (std + 1e-9)




def compute_and_normalize_fft(output, target):

    # Compute FFT on the last dimension of both output and target
    # target_fft = torch.fft.rfft(target, dim=-1)
    # output_fft = torch.fft.rfft(output, dim=-1)

    # target_fft = torch.fft.rfft(target.float(), dim=-1)
    # output_fft = torch.fft.rfft(output.float(), dim=-1)

    # target = pad_to_power_of_two(target)
    # output = pad_to_power_of_two(output)

    target_fft = torch.fft.rfft(target, dim=-1)
    output_fft = torch.fft.rfft(output, dim=-1)

    # Calculate amplitude and phase
    target_amp = torch.abs(target_fft)
    target_phase = torch.angle(target_fft)
    output_amp = torch.abs(output_fft)
    output_phase = torch.angle(output_fft)

    # Z-score normalization
    target_amp_norm = z_score_normalize(target_amp)
    target_phase_norm = z_score_normalize(target_phase)
    output_amp_norm = z_score_normalize(output_amp)
    output_phase_norm = z_score_normalize(output_phase)

    return target_amp_norm, target_phase_norm, output_amp_norm, output_phase_norm


def fft_masked_mse_loss(output, target, mask, current_batch,  phase_start_batch=250, phase_end_batch=500, normalize_fft=False, channel_index=0, patch_index=0, visualize=True, include_phase=False, frequency_range=None):
# def fft_masked_mse_loss(output, target, mask,current_batch, start_batch, end_batch, normalize_fft=False, channel_index=0, patch_index=0, visualize=True, include_phase=False, frequency_range=None):
    """
    Computes masked MSE loss between target and output using their FFT magnitudes, and visualizes the FFT of a specified patch and channel.

    Args:
        output (Tensor): The output tensor from the model.
        target (Tensor): The target tensor.
        mask (Tensor): The mask tensor indicating valid data points.
        channel_index (int): The index of the channel to visualize.
        patch_index (int): The index of the patch to visualize.
        frequency_range (tuple or None): Frequency range for visualization.
    Returns:
        Tensor: The computed loss.
    """

    # Compute FFT and normalize
    target_amp_norm, target_phase_norm, output_amp_norm, output_phase_norm = compute_and_normalize_fft(output, target)

    mask_fft = mask.unsqueeze(-1)  # Add an extra dimension for broadcasting

    # Loss for amplitude
    target_amp_masked = target_amp_norm * mask_fft
    output_amp_masked = output_amp_norm * mask_fft

    loss_amp = F.mse_loss(output_amp_masked, target_amp_masked, reduction='sum') / (mask_fft.sum() + 0.000001)

    if include_phase:
        # Loss for phase
        target_phase_masked = target_phase_norm * mask_fft
        output_phase_masked = output_phase_norm * mask_fft
        loss_phase = F.mse_loss(output_phase_masked, target_phase_masked, reduction='sum') / (mask_fft.sum() + 0.000001)

        print("current_batch, phase_start_batch, phase_end_batch", current_batch, phase_start_batch, phase_end_batch)

        phase_weight = calculate_phase_weight(current_batch, phase_start_batch, phase_end_batch)
        print("Phase weight", phase_weight)
        loss_phase = loss_phase * phase_weight

        print("loss_amp", loss_amp)
        print("loss_phase", loss_phase)


        # Combine losses
        total_loss = loss_amp + loss_phase



        del target_phase_masked, output_phase_masked, loss_phase, phase_weight
    else:
        total_loss = loss_amp

    if visualize:
        visualize_fft(target_amp_norm[patch_index, channel_index, :], target_phase_norm[patch_index, channel_index, :], output_amp_norm[patch_index, channel_index, :], output_phase_norm[patch_index, channel_index, :], frequency_range)

    del output, target, mask, loss_amp
    return total_loss

    # # Use only the magnitude for training
    # target_fft_mag = torch.abs(target_amp_norm)
    # output_fft_mag = torch.abs(output_amp_norm)

    # # print("target_fft_mag.shape,  output_fft_mag.shape", target_fft_mag.shape,  output_fft_mag.shape)

    # # print("mask,", mask.shape)


    # # Apply the mask to the magnitude of FFT-transformed signals
    # mask_fft = mask.unsqueeze(-1)  # Add an extra dimension for broadcasting


    # target_masked = target_fft_mag * mask_fft
    # output_masked = output_fft_mag * mask_fft

    # # print("mask_fft", mask_fft.shape)


    # # Calculate MSE loss only on the non-padded (real) regions
    # epsilon = 0.000001 # Small value to prevent division by zero
    # loss = F.mse_loss(output_masked, target_masked, reduction='sum') / (mask_fft.sum() + epsilon)

    # # patch_index = random.randint(0, 900)
    # # channel_index = random.randint(0,18)

    # print(target_amp_norm[patch_index, channel_index, :].shape, target_phase_norm[patch_index, channel_index, :].shape, output_amp_norm[patch_index, channel_index, :].shape, output_phase_norm[patch_index, channel_index, :].shape)

    # visualize_fft(target_amp_norm[patch_index, channel_index, :], target_phase_norm[patch_index, channel_index, :], output_amp_norm[patch_index, channel_index, :], output_phase_norm[patch_index, channel_index, :], frequency_range)

    # del output, target, mask, target_masked, target_fft_mag, output_fft_mag
    # return loss
