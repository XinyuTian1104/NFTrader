import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TimeSeriesEncoder(nn.Module):
    def __init__(self, num_features, num_hidden=256, num_layers=6, num_heads=8, output_size=2048):
        super(TimeSeriesEncoder, self).__init__()

        self.num_hidden = num_hidden

        self.position_encoding = nn.Parameter(torch.randn(1, 1, num_hidden))
        self.feature_embedding = nn.Linear(num_features, num_hidden)

        self.transformer_layer = TransformerEncoderLayer(
            d_model=num_hidden,
            nhead=num_heads,
            dim_feedforward=num_hidden * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.transformer_layer,
            num_layers=num_layers
        )

        self.projection_layer = nn.Linear(num_hidden, output_size)

    def forward(self, x):
        # print("size:", x.size())
        batch_size, timesteps, _ = x.size()
        

        x = self.feature_embedding(x)
        pos_enc = self.position_encoding.repeat(1, timesteps, 1)
        x = x + pos_enc

        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.projection_layer(x)

        return x


def test_ts_encoder():
    num_features = 20
    model = TimeSeriesEncoder(num_features=num_features)
    # print(model)
    input_tensor = torch.randn(32, 500, num_features)
    print("TS Input shape:", str(input_tensor.shape))
    output = model(input_tensor)
    print(output.shape)
