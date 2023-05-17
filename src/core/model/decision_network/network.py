import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class DecisionNetwork(nn.Module):
    def __init__(self, encode_feature_dim, ts_timestep, ts_dim, manual_feature_dim, n_classes=3):
        super(DecisionNetwork, self).__init__()

        self.transformer_encoder_layer = TransformerEncoderLayer(
            d_model=ts_dim, nhead=4)
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layer, num_layers=2)

        self.fc1 = nn.Linear(encode_feature_dim +
                             ts_dim + manual_feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)
        # Additional output for transaction percentage
        self.fc4 = nn.Linear(128, 1)

    def forward(self, encode_features, ts_data, manual_features):
        # Apply the transformer encoder to time series data
        transformer_output = self.transformer_encoder(ts_data)

        # Get the last time step output
        last_ts_output = transformer_output[:, -1, :]

        # Concatenate the three inputs
        combined_features = torch.cat(
            (encode_features, last_ts_output, manual_features), dim=1)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        # Output percentage using sigmoid activation function
        percentage = torch.sigmoid(self.fc4(x))

        return actions, percentage


def test_decision_network():
    # Sample data
    batch_size = 32
    encode_feature_dim = 2048 * 3
    ts_timestep = 64
    ts_dim = 8
    manual_feature_dim = 16

    encode_features = torch.randn(batch_size, encode_feature_dim)
    ts_data = torch.randn(batch_size, ts_timestep, ts_dim)
    manual_features = torch.randn(batch_size, manual_feature_dim)

    decision_network = DecisionNetwork(
        encode_feature_dim, ts_timestep, ts_dim, manual_feature_dim)

    # Process sample data
    output_actions, output_percentage = decision_network(
        encode_features, ts_data, manual_features)

    # Get the most likely action
    predicted_actions = torch.argmax(output_actions, dim=1)

    # Calculate the amount to buy, sell, or hold
    current_holdings = 100
    transaction_amount = (output_percentage * current_holdings).squeeze()

    print("Predicted actions:", predicted_actions)
    print("Transaction percentage:", output_percentage)
    print("Transaction amount:", transaction_amount)

    return
