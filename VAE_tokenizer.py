
class TemporalEncoder(nn.Module):
    def __init__(self, in_channels=1, conv_out_channels=8, emb_dim=64, patch_size=200):
        super(TemporalEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
        )

        conv_output_size = 64

        self.proj_to_emb = nn.Linear(conv_output_size, emb_dim)

    def forward(self, x):
        # x shape: [batch, eeg_channel, patches, length]
        batch_size, eeg_channel, patches, length = x.size()
        # Correct reshaping: treat each patch within each EEG channel independently
        x = x.view(batch_size * eeg_channel * patches, 1, 1, length)

        # print(("x before convolution", x.shape))

        x = self.conv_layers(x)

        x = x.mean(dim=[2, 3])  # This is just an example; adjust based on your architecture

        x = self.proj_to_emb(x)
        # Reshape back to [batch, eeg_channel, patches, emb_dim]
        x = x.view(batch_size, eeg_channel, patches, -1)

        return x

class TemporalDecoder(nn.Module):
    def __init__(self, in_channels=64, conv_out_channels=8, emb_dim=64, patch_size=200):
        super(TemporalDecoder, self).__init__()

        self.proj_from_emb = nn.Linear(64, 64)  # Adjusted for clarity


        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7), output_padding=(0, 1))
        )
        self.to_single_channel = nn.Conv2d(8, 1, kernel_size=(1, 1))

        # Adaptive pooling layer to ensure the output is [1, 200] for each patch
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, patch_size))


    def forward(self, x):
            batch_size, eeg_channel, patches, _ = x.size()
            x_flattened = x.view(batch_size * eeg_channel * patches, -1)
            x_projected = self.proj_from_emb(x_flattened)

            # Correct reshaping to align with deconv layers' expected input
            x_reshaped = x_projected.view(batch_size * eeg_channel * patches, 1, 1, 64)

            x = self.deconv_layers(x_reshaped)

            # Apply adaptive pooling here to resize to the target shape
            x = self.adaptive_pooling(x)

            x = self.to_single_channel(x)

            return x

class VQVAEDecoder(nn.Module):
    def __init__(self, codebook_size=1024, emb_dim=64, patch_size=200):
        super(VQVAEDecoder, self).__init__()
        self.temporal_decoder = TemporalDecoder(emb_dim=emb_dim, patch_size=patch_size)
        # Additional components such as the transformer can be mirrored here if necessary for the decoding process

    def forward(self, x):
        # Decode each patch from embeddings
        decoded = self.temporal_decoder(x)  # Shape: [batch, eeg_channel, patches, length]

        return decoded


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_embedding', self.embedding.weight.data.clone())

    def forward(self, z):

        # z is the continuous embedding produced by the decoder

        original_shape = z.shape[:-1]  # Capture the original shape, excluding the embedding dimension, to be used for arranging the token tensor
        # Flatten z to fit with the embedding shape
        z_flattened = z.view(-1, self.embedding_dim)

        # Calculate distances between z and every entry in the codebook
        distances = (
            z_flattened.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        closest_indices = distances.argmin(1)
        closest_embeddings = self.embedding(closest_indices)

        if self.training:
        # Update EMA variables
            with torch.no_grad():
                    self.ema_cluster_size *= self.decay
                    self.ema_cluster_size.index_add_(0, closest_indices, (1-self.decay)*torch.ones_like(closest_indices, dtype=torch.float))

                    n = self.ema_cluster_size.sum()
                    self.ema_cluster_size = (
                        (self.ema_cluster_size + self.epsilon)
                        / (n + self.num_embeddings * self.epsilon) * n
                    )


            dw = torch.matmul(z_flattened.t(), one_hot(closest_indices, self.num_embeddings))
            self.ema_embedding *= self.decay
            self.ema_embedding += (1 - self.decay) * dw.t()

            # Normalization step to ensure numerical stability
            self.embedding.weight = nn.Parameter(self.ema_embedding / self.ema_cluster_size.unsqueeze(1))

        # Reshape closest_embeddings to the original input shape
        quantized = closest_embeddings.view_as(z)

        # After obtaining closest_indices, reshape them back to the structured format
        structured_indices = closest_indices.view(original_shape)

        return quantized, structured_indices

def one_hot(indices, num_classes):
    # Move the eye tensor to the same device as indices
    return torch.eye(num_classes, device=indices.device)[indices]


class VQVAE_v1_w_codebook(nn.Module):
    def __init__(self, codebook_size=1024, emb_dim=64,  patch_size=200, quantize=True, use_saved_codebook=False):
        super(VQVAE_v1_w_codebook, self).__init__()
        self.temporal_encoder = TemporalEncoder(emb_dim=emb_dim)
        self.decoder = VQVAEDecoder(emb_dim=emb_dim, patch_size=patch_size)
        self.quantizer = VectorQuantizerEMA(num_embeddings=codebook_size, embedding_dim=emb_dim, decay=0.99)
        self.quantize = quantize


        print("Instantiating VQVAE_v1_w_codebook")
        print("Quantize: ", quantize)
        print("codebook_size", codebook_size)

        # # Initialize codebook based on use_saved_codebook flag
        # if use_saved_codebook and codebook_tensor is not None:
        #     # Use the provided saved codebook tensor
        #     assert codebook_tensor.size() == (codebook_size, emb_dim), "Codebook tensor dimensions do not match."
        #     self.quantizer.embedding.weight.data.copy_(codebook_tensor)
        #     print("Using saved codebook for initialization.")
        # else:
        #     # Randomly initialize the codebook
        #     self.quantizer.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        #     print("Randomly initializing the codebook.")


    def forward(self, x):

        encoded = self.temporal_encoder(x)

        # if self.quantize:
        #     quantized, token_indices = self.quantizer(encoded)
        #     decoded = self.decoder(quantized)

        # else:
        #     decoded = self.decoder(encoded)
        #     quantized = decoded
        #     token_indices = None

        if self.training:
            decoded = self.decoder(encoded)
            batch_size, EEG_channels, num_patches, _ = x.shape
            decoded = decoded.view(batch_size, EEG_channels, num_patches, -1)
            quantized, token_indices = self.quantizer(encoded)

            return quantized, decoded, token_indices

        if not self.training:
            with torch.no_grad():
                quantized, token_indices = self.quantizer(encoded)
                decoded = None

                return quantized, decoded, token_indices

