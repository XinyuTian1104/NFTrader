import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision import transforms


class ImageTransformerEncoder(nn.Module):
    def __init__(self, feature_dim, num_layers, num_heads, d_ff, dropout):
        super(ImageTransformerEncoder, self).__init__()

        # Pre-trained image feature extraction model (ResNet-50)
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Max pool for fixed-length feature vector
        self.max_pool = nn.AdaptiveMaxPool2d((1, feature_dim))

    def forward(self, x):
        # Extract image features
        features = self.feature_extractor(x)
        batch_size, _, H, W = features.size()

        # Flatten and permute dimensions
        features = features.view(batch_size, -1, H * W).permute(2, 0, 1)

        # Apply transformer layers
        for layer in self.transformer_layers:
            features = layer(features)

        # Create fixed-length feature vector
        features = features.permute(1, 2, 0)
        feature_vector = self.max_pool(features).squeeze(2)

        return feature_vector


class ViTEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, patch_size, feature_dim=2048):
        super(ViTEncoder, self).__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers)

        self.position_encoding = nn.Parameter(
            torch.randn(1, (patch_size * patch_size) + 1, d_model))

        self.max_pool = nn.AdaptiveMaxPool2d((1, feature_dim))

    def forward(self, x):
        # Flatten and add position encoding
        x = x.view(x.shape[0], self.d_model, -1).permute(2, 0, 1)
        x = x + self.position_encoding[:, :x.size(0), :].detach()

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Get the [CLS] token (first token)
        x = x[0]

        # map to feature dimension
        x = self.max_pool(x.unsqueeze(0)).squeeze(0)

        return x


def test_vit():
    # Parameters
    image_path = '/Users/crinstaniev/Courses/STATS402/data/raw/0n1-force/image.png'
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    patch_size = 16

    # Load and preprocess image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Instantiate the model
    model = ViTEncoder(d_model, nhead, num_layers, dim_feedforward, patch_size)
    print(model)

    with torch.no_grad():
        result = model(image_tensor)

    print(result.size())
