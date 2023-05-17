import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from core.model.encoders.multimodal_encoder import MultimodalEncoder


class DecisionNetwork(nn.Module):
    def __init__(self, encode_feature_dim, ts_dim, n_classes=3):
        super(DecisionNetwork, self).__init__()

        self.multimodal_encoder = MultimodalEncoder(
            ts_num_features=5,
        )

        multimodal_encoder_output_size = 2048 + 2048 + 2048

        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=ts_dim, nhead=5)
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layer, num_layers=2)

        self.fc1 = nn.Linear(encode_feature_dim +
                             ts_dim + multimodal_encoder_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        # Additional output for transaction percentage
        self.fc4 = nn.Linear(128, 1)

    def forward(self,
                ts_features,
                image_features,
                text_features,
                encode_features,
                ts_data):

        # multimodal encoder
        multimodal_encoder_output = self.multimodal_encoder(
            ts_features, image_features, text_features)

        # Apply the transformer encoder to time series data
        transformer_output = self.transformer_encoder(ts_data)

        # Get the last time step output
        last_ts_output = transformer_output[:, -1, :]

        # Concatenate the three inputs
        combined_features = torch.cat(
            (encode_features, last_ts_output, multimodal_encoder_output), dim=1)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        # Output percentage using sigmoid activation function
        percentage = torch.sigmoid(self.fc4(x))

        return actions, percentage


def test_decision_network():
    # Sample data
    batch_size = 1
    encode_feature_dim = 2048 * 3
    ts_dim = 5
    ts_timestep = 8

    # (batch_size, channels, height, width)
    image_sample = torch.randn(1, 3, 224, 224)
    text_sample = [
        'This is a sample text to test the MultimodalEncoder module.'
        for _ in range(1)
    ]  # (batch_size, x)
    # (batch_size, timesteps, num_features)
    ts_sample = torch.randn(1, 16, 5)

    encode_features = torch.randn(batch_size, encode_feature_dim)
    ts_data = torch.randn(batch_size, ts_timestep, ts_dim)

    decision_network = DecisionNetwork(
        encode_feature_dim, ts_dim)

    # Process sample data
    output_actions, output_percentage = decision_network(
        ts_sample, image_sample, text_sample,
        encode_features, ts_data)

    # Get the most likely action
    predicted_actions = torch.argmax(output_actions, dim=1)

    # Calculate the amount to buy, sell, or hold
    current_holdings = 100
    transaction_amount = (output_percentage * current_holdings).squeeze()

    print("Predicted actions:", predicted_actions)
    print("Transaction percentage:", output_percentage)
    print("Transaction amount:", transaction_amount)

    return
