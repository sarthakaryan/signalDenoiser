import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len // patch_size, embed_dim))

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 1)  # (batch_size, num_patches, embed_dim)
        x += self.pos_embed
        return x

class ExplainableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ExplainableMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1,padding=1)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        reduced_x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Ensure the reduced_x has the same sequence length
        key = reduced_x
        query = reduced_x
        value = x

        attn_output, _ = self.attention(query, key, value)
        return attn_output

class DualPhaseTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super(DualPhaseTransformerBlock, self).__init__()
        self.mce_msa = ExplainableMultiHeadAttention(embed_dim, num_heads)
        self.gru = nn.GRU(embed_dim, ffn_dim, batch_first=True)
        self.fc = nn.Linear(ffn_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output = self.mce_msa(x)
        x = self.norm1(x + attn_output)
        gru_output, _ = self.gru(x)
        ffn_output = F.relu(self.fc(gru_output))
        x = self.norm2(x + ffn_output)
        return x

class DPATD(nn.Module):
    def __init__(self, seq_len, patch_size, embed_dim, num_heads, ffn_dim, num_layers):
        super(DPATD, self).__init__()
        self.embed = PatchEmbedding(seq_len, patch_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            DualPhaseTransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.deconv = nn.ConvTranspose1d(embed_dim, 1, kernel_size=patch_size, stride=patch_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.embed(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = self.deconv(x)
        return x

    def compute_loss(self, clean_audio, predicted_audio):
        return self.loss_fn(predicted_audio, clean_audio)

def train_dpatd(model, noisy_audio, clean_audio, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        denoised_audio = model(noisy_audio)
        loss = model.compute_loss(clean_audio, denoised_audio)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


