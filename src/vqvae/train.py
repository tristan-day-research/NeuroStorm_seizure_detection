
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split
from functools import partial

def adjust_loss_weights(batch_idx, start_batch, end_batch, init_recon, final_recon, init_quant, final_quant):
    if batch_idx < start_batch:
        return init_recon, init_quant
    if batch_idx > end_batch:
        return final_recon, final_quant

    progress = (batch_idx - start_batch) / max(1, end_batch - start_batch)
    recon_weight = max(0.1, init_recon + progress * (final_recon - init_recon))
    quant_weight = min(1e6, init_quant + progress * (final_quant - init_quant))
    return recon_weight, quant_weight


def train(model, train_loader, optimizer, loss_function, device, scheduler=None, accum_steps=1):
    model.train()
    total_loss_tracker = 0
    global total_train_batches

    for batch_idx, (file_name, eeg_tensor, mask) in enumerate(train_loader):
        if batch_idx > max_batches_per_train_epoch - 2:
            break

        torch.cuda.empty_cache()
        optimizer.zero_grad()
        target, mask = eeg_tensor.to(device), mask.to(device)

        embedding, decoded = model(target)
        recon_loss = loss_function(target, decoded, mask, batch_idx)

        recon_weight, quant_weight = adjust_loss_weights(total_train_batches,
                                                         start_batch_for_adjustment,
                                                         end_weight_transfer_batch,
                                                         initial_recon_weight,
                                                         final_recon_weight,
                                                         initial_quant_weight,
                                                         final_quant_weight)

        quantized, _ = model.quantizer(embedding)
        quant_loss = torch.mean((quantized.detach() - embedding)**2) + \
                     beta * torch.mean((quantized - embedding.detach())**2)

        total_loss = recon_weight * recon_loss + quant_weight * quant_loss
        total_loss.backward()
        optimizer.step()

        total_losses.append(total_loss.item())
        reconstruction_losses.append(recon_loss.item())
        quantization_losses.append(quant_loss.item())

        if scheduler:
            scheduler.step()
        total_train_batches += 1

    return total_loss_tracker / len(train_loader)


def validate(model, valid_loader, loss_function, device, max_batches=None):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, mask) in enumerate(valid_loader):
            if batch_idx > max_batches_per_val_epoch - 2:
                break
            target, mask = data.to(device), mask.to(device)
            embedding, decoded = model(target)
            recon_loss = loss_function(target, decoded, mask, batch_idx)
            quantized, _ = model.quantizer(embedding)
            quant_loss = torch.mean((quantized.detach() - embedding)**2) + \
                         beta * torch.mean((quantized - embedding.detach())**2)
            total_loss = recon_loss + quantization_loss_weight * quant_loss
            total_val_loss += total_loss.item()

    return total_val_loss / len(valid_loader)





# from torch.utils.data import DataLoader, Subset, random_split
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# # torch.cuda.memory_summary()


# # Hyperparameters for data loader

# stride = 150
# batch_size = 4
# num_workers = 4


# # Hyperparameters for model

# model_name = "VQVAE_v1"
# codebook_size = 1024
# emb_dim = 64
# codebook_tensor = None
# codebook_training = True
# use_saved_codebook = False

# # Training Hyperparameters

# num_epochs = 7
# lr = 1e-4
# lr_scheduler_step_size = 1
# lr_scheduler_gamma = 0.9
# accumulation_steps = 2
# quantization_loss_weight = 1  # This is only used in the validation funciton
# # weight_decay = 1e-3  # For AdamW optimizer


# max_batches_per_train_epoch = 50  # Define maximum number of batches per epoch
# max_batches_per_val_epoch = 6

# # Loss

# visualize_the_fft = False

# start_batch_for_adjustment = 100
# end_weight_transfer_batch = 500

# initial_recon_weight =1.0
# final_recon_weight = 0.9

# initial_quant_weight = 0.0
# final_quant_weight = 500000 #  1 with 5 0s

# beta = 0.25    #Commitment Cost: beta acts as a weighting factor for the second term of the
#                     #  quantization loss, often referred to as the "commitment cost". This cost encourages
#                     #  the encoder's outputs (encoded) to be close to one of the vectors in the codebook (quantized),
#                     #  essentially committing the encoder to the chosen codebook vector.




# include_phase = False
# phase_start_batch = 200
# phase_end_batch = 400



# # Probably won't adjust these
# blob_prefix = "train_eegs_HMS_processed"
# patch_size = 200
# max_patches = 1000
# n_channels = 19

# loss_function = partial(fft_masked_mse_loss, phase_start_batch=phase_start_batch, phase_end_batch=phase_end_batch, normalize_fft=False, channel_index=0, patch_index=0, visualize=visualize_the_fft, include_phase=include_phase, frequency_range=None)


# # Clear memory
# torch.cuda.empty_cache()
# import gc
# # If you have an existing model called 'model', delete it
# if 'model' in locals():
#     del model
#     torch.cuda.empty_cache()  # Clears cached memory
#     gc.collect()  # Explicitly invokes the garbage collection

# # Initialize dataset and data loader
# transform_to_patches = ToPatches(patch_size=patch_size, stride=stride, max_patches=max_patches)
# eeg_dataset = EEGDataset(bucket_name=BUCKET_NAME, blob_prefix=blob_prefix, transform=transform_to_patches)
# data_loader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# total_size = len(eeg_dataset)
# train_size = int(0.8 * total_size)  # 80% for training
# valid_size = total_size - train_size  # Remaining 20% for validation
# train_dataset, valid_dataset = random_split(eeg_dataset, [train_size, valid_size])

# # Optionally, create a smaller training subset for faster epochs (e.g., 50% of train_dataset)
# subset_size = int(0.5 * len(train_dataset))  # Adjust as necessary
# indices = torch.randperm(len(train_dataset))[:subset_size]
# train_subset = Subset(train_dataset, indices)

# train_loader = DataLoader(train_subset, batch_size=batch_size , shuffle=True,  num_workers=num_workers)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



# # Get the class object using globals()
# model_class = globals()[model_name]
# model = model_class(codebook_size=codebook_size, emb_dim=emb_dim, quantize=False)
# model.to(device)

# weight_decay = 1e-3  # You can adjust this value as recommended
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# scaler = GradScaler()


# # Initialize the scheduler
# scheduler = StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

# def adjust_loss_weights(batch_idx, start_batch, end_weight_transfer_batch, initial_recon_weight, final_recon_weight, initial_quant_weight, final_quant_weight):
#     """
#     Calculate the current reconstruction and quantization loss weights based on the training progress.
#     The function ensures that current_recon_weight does not fall below 0.1 and current_quant_weight does not exceed 10000.
#     """
#     if batch_idx < start_batch:
#         return initial_recon_weight, initial_quant_weight

#     if batch_idx > end_weight_transfer_batch:
#         return final_recon_weight, final_quant_weight

#     progress = (batch_idx - start_batch) / max(1, end_weight_transfer_batch - start_batch)  # Avoid division by zero

#     current_recon_weight = initial_recon_weight + progress * (final_recon_weight - initial_recon_weight)
#     current_quant_weight = initial_quant_weight + progress * (final_quant_weight - initial_quant_weight)

#     # Clamp the weights to their specified minimum or maximum values
#     current_recon_weight = max(0.1, current_recon_weight)
#     current_quant_weight = min(1000000, current_quant_weight)

#     return current_recon_weight, current_quant_weight


# avg_val_losses_list = []

# def validate(model, valid_loader, loss_function, device, max_batches=None):
#     model.eval()  # Set the model to evaluation mode
#     total_val_loss = 0
#     batches_processed = 0

#     print("++++++___________VALIDATION___________++++++++")

#     with torch.no_grad():  # No need to track gradients for validation
#         for batch_idx, (data, mask) in enumerate(valid_loader):
#             if batch_idx > max_batches_per_val_epoch -2:
#                 break

#             target, mask = data.to(device), mask.to(device)
#             embedding, decoded = model(target)
#             reconstruction_loss = loss_function(target, decoded, mask, batch_idx)
#             # Get quantized embeddings
#             quantized, _ = model.quantizer(embedding)
#             quantization_loss = torch.mean((quantized.detach() - embedding)**2) + beta * torch.mean((quantized - embedding.detach())**2)

#             total_loss = reconstruction_loss + quantization_loss_weight * quantization_loss
#             total_val_loss += total_loss.item()

#             batches_processed += 1

#     avg_val_loss = total_val_loss / min(len(valid_loader), batches_processed)
#     avg_val_losses_list.append(avg_val_loss)
#     plot_per_batch("avg_val_losses_list", avg_val_losses_list)
#     print("avg_val_loss", avg_val_loss)
#     return avg_val_loss


# # WITHOUT MIXED PRECISION TRAINING OR GRADIENT ACCUMULATION
# # Pad to the power of 2 should be off

# total_train_batches = 0
# total_losses = []
# reconstruction_losses = []
# quantization_losses = []
# unweighted_reconstruction_losses = []
# unweighted_quantization_losses = []

# new_lr = lr

# # def train(model, train_loader, optimizer, criterion, device, scheduler=None, accumulation_steps=4):
# def train(model, train_loader, optimizer, loss_function, device, codebook_training=True, scheduler=None, accumulation_steps=accumulation_steps):
#     model.train()
#     total_loss_tracker = 0
#     global total_train_batches
#     global new_lr

#     for batch_idx, (file_name, eeg_tensor, mask) in enumerate(train_loader):
#         if batch_idx > max_batches_per_train_epoch - 2:
#             break
#         print("batch", batch_idx)
#         print("total_train_batches", total_train_batches)
#         torch.cuda.empty_cache()
#         optimizer.zero_grad()

#         # if eeg_tensor is None or any(d is None for d in data):
#         #     print(f"Skipping batch {batch_idx} due to None data")
#         #     continue  # Skip this batch

#         target, mask = eeg_tensor.to(device), mask.to(device)


#         if codebook_training:

#             # # Forward pass through the model
#             embedding, decoded = model(target)  # Here 'model' should return both decoded (reconstructed inputs) and encoded (embeddings) outputs

#             # # Compute the reconstruction loss
#             reconstruction_loss = loss_function(target, decoded, mask, batch_idx)

#             # Adjust loss weights based on the batch index

#             recon_weight, quant_weight = adjust_loss_weights(total_train_batches, start_batch_for_adjustment, end_weight_transfer_batch,  initial_recon_weight, final_recon_weight, initial_quant_weight, final_quant_weight)

#             # Get quantized embeddings
#             quantized, _ = model.quantizer(embedding)
#             quantization_loss = torch.mean((quantized.detach() - embedding)**2) + beta * torch.mean((quantized - embedding.detach())**2)

#             weighted_recon_loss = recon_weight * reconstruction_loss
#             weighted_quant_loss = quant_weight * quantization_loss

#              # Apply the dynamically adjusted weights
#             total_loss =  weighted_recon_loss + weighted_quant_loss

#             print("recon_weight, quant_weight ", recon_weight, quant_weight )

#         else:

#             embedding, decoded = model(target)
#             weighted_recon_loss = loss_function(target, decoded, mask, batch_idx)
#             weighted_quant_loss = torch.tensor([0])

#         print("current lr after scheduler step", new_lr)
#         print("UNweighted reconstruction_loss", reconstruction_loss.cpu().detach())
#         print("UNweighted quantization_loss", quantization_loss.cpu().detach())
#         print("weighted reconstruction_loss", weighted_recon_loss.cpu().detach())
#         print("weighted quantization_loss", weighted_quant_loss.cpu().detach())

#         print("total_loss", total_loss.cpu().detach())

#         total_loss.backward()
#         optimizer.step()

#         # Append losses to lists for tracking and then plot them
#         total_losses.append(total_loss.item())
#         reconstruction_losses.append(weighted_recon_loss.item())
#         quantization_losses.append(weighted_quant_loss.item())
#         unweighted_reconstruction_losses.append(reconstruction_loss.cpu().detach().item())
#         unweighted_quantization_losses.append(quantization_loss.cpu().detach().item())

#         plot_per_batch("UNweighted reconstruction_loss", unweighted_reconstruction_losses)
#         # plot_per_batch("UNweighted quantization_loss", unweighted_quantization_losses)
#         plot_per_batch("weighted reconstuction_loss", reconstruction_losses)
#         plot_per_batch("weighted quantization_loss", quantization_losses)
#         plot_per_batch(f"Total weighted loss for batch {total_train_batches}", total_losses)


#         total_loss_tracker += total_loss.item()

#         if batch_idx % 10 == 0:  # Adjust the frequency as needed
#             print(f"Batch {batch_idx}, Loss: {total_loss.item()}")


#         total_train_batches += 1

#         del target, mask, total_loss, embedding, decoded, eeg_tensor, token_indices, quantized

#     avg_loss = total_loss_tracker / len(train_loader)
#     print(f"+++____________Epoch Completed. Average Loss: {avg_loss:.4f}_________+++")
#     if scheduler:
#         scheduler.step()
#         new_lr = optimizer.param_groups[0]['lr']
#         print(f"New learning rate: {new_lr}")

#     return avg_loss


# model.train()

# for epoch in range(num_epochs):
#     print(f"+++++___________________EPOCH {epoch+1}/{num_epochs}_____________________++++++++++++++++++++")
#     avg_loss = train(model, train_loader, optimizer, loss_function, device, codebook_training=codebook_training, scheduler=scheduler, accumulation_steps=accumulation_steps)

#     print(f"+++____EPOCH {epoch+1}, Average Loss: {avg_loss:.4f}_____++++++++++++++++")
#     avg_val_loss = validate(model, valid_loader, loss_function, device)



