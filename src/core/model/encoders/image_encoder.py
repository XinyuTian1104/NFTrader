import torch
import torch.nn as nn


class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size, emb_size, num_heads, num_layers, feature_dim, dropout, channel):
        super(ViTEncoder, self).__init__()

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        # patch_dim = channel * patch_size ** 2

        self.patch_embedding = nn.Conv2d(
            channel, emb_size, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                emb_size, num_heads, dim_feedforward=4 * emb_size, dropout=dropout), num_layers
        )
        self.fc = nn.Linear(emb_size, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.emb_size = emb_size

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding[:, :(self.num_patches + 1)]
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


def test_vit():
    image_size = 224
    patch_size = 16
    emb_size = 256
    num_heads = 8
    num_layers = 12
    feature_dim = 2048
    # feature_dim = 128
    dropout = 0.1
    channel = 3

    model = ViTEncoder(image_size, patch_size, emb_size,
                       num_heads, num_layers, feature_dim, dropout, channel)

    batch_size = 32
    input_images = torch.randn(batch_size, channel, image_size, image_size)

    print("input shape: ", input_images.shape)

    output = model(input_images)

    print("output shape: ", output.shape)
