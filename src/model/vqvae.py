class Encoder(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, embedding_dim, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_dim, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_dim, 1, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x
    
    # @title Solution 👀

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        # z: (batch, channel, height, width)
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)

        # Compute distances
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t()))

        # Find nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)  # push encoder outputs closer to embeddings
        q_latent_loss = F.mse_loss(quantized, z.detach())  # improve embeddings to match encoder outputs
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, loss
    
    # Full VQ-VAE Model

class VQVAE(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
    #def __init__(self, hidden_dim=64, embedding_dim=32, num_embeddings=128, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(hidden_dim, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss