import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from core.model.encoders.multimodal_encoder import MultimodalEncoder

VECTOR_LENGTH = 16 * 5 + 3 * 224 * 224 + 16 * 5 + 1 + 1


class DecisionNetwork(nn.Module):
    def __init__(self, ts_dim=5, n_classes=3):
        super(DecisionNetwork, self).__init__()

        self.multimodal_encoder = MultimodalEncoder(
            ts_num_features=5,
        )

        multimodal_encoder_output_size = 128 * 2

        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=ts_dim, nhead=5)
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layer, num_layers=2)

        self.fc1 = nn.Linear(ts_dim + multimodal_encoder_output_size + 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x):
        # if x is not a tensor, convert it to a tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x).float()

        # shape of x: (batch_size, VECTOR_LENGTH)
        # split x into different features

        # shape of x: (batch_size, VECTOR_LENGTH)
        # structure of x:
        #  - ts_feature: (batch_size, 16, 5)
        #  - image_feature: (batch_size, 3, 224, 224)
        #  - ts_data: (batch_size, 16, 5)
        #  - usd_wallet: (batch_size, 1)
        #  - nft_wallet: (batch_size, 1)
        # split x into different features
        ts_features = x[:, :16 * 5].reshape(-1, 16, 5)
        image_features = x[:, 16 * 5:16 *
                           5 + 3 * 224 * 224].reshape(-1, 3, 224, 224)
        ts_data = x[:, 16 * 5 + 3 * 224 *
                    224:16 * 5 + 3 * 224 * 224 + 16 * 5].reshape(-1, 16, 5)
        usd_wallet = x[:, 16 * 5 + 3 * 224 * 224 + 16 * 5].reshape(-1, 1)
        nft_wallet = x[:, 16 * 5 + 3 *
                       224 * 224 + 16 * 5 + 1].reshape(-1, 1)

        # multimodal encoder
        multimodal_encoder_output = self.multimodal_encoder(
            ts_features, image_features)

        # Apply the transformer encoder to time series data
        transformer_output = self.transformer_encoder(ts_data)

        # Get the last time step output
        last_ts_output = transformer_output[:, -1, :]

        # Concatenate the three inputs
        combined_features = torch.cat(
            (last_ts_output,
             multimodal_encoder_output,
             usd_wallet,
             nft_wallet),
            dim=1)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


def test_decision_network():
    # Sample data
    batch_size = 32
    ts_dim = 5

    decision_network = DecisionNetwork(ts_dim)

    sample_data = torch.randn(size=(batch_size, VECTOR_LENGTH))

    # Process sample data
    output_actions = decision_network(sample_data)

    print('output_actions: ', output_actions)

    # Check output shape
    print('output_actions.shape: ', output_actions.shape)

    return
