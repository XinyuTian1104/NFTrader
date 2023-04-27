import torch
import torch.nn as nn
from core.model.encoders.image_encoder import ViTEncoder
from core.model.encoders.text_encoder import BertEncoder
from core.model.encoders.ts_encoder import TimeSeriesEncoder

class MultimodalEncoder(nn.Module):
    def __init__(self, ts_num_features, img_d_model=512, img_nhead=8, img_num_layers=6, img_dim_feedforward=2048, img_output_size=2048, text_hidden_size=768, text_output_size=2048, ts_num_hidden=256, ts_num_layers=6, ts_num_heads=8, ts_output_size=2048):
        super(MultimodalEncoder, self).__init__()

        self.image_encoder = ViTEncoder(
            d_model=img_d_model,
            nhead=img_nhead,
            num_layers=img_num_layers,
            dim_feedforward=img_dim_feedforward,
            output_size=img_output_size
        )
        self.text_encoder = BertEncoder(
            hidden_size=text_hidden_size,
            output_size=text_output_size
        )
        self.ts_encoder = TimeSeriesEncoder(
            num_features=ts_num_features,
            num_hidden=ts_num_hidden,
            num_layers=ts_num_layers,
            num_heads=ts_num_heads,
            output_size=ts_output_size
        )

    def forward(self, time_series, image, text):
        time_series = self.ts_encoder(time_series)
        image = self.image_encoder(image)
        text = self.text_encoder(text)

        combined = torch.cat([time_series, image, text], dim=1)

        return combined


def test_multimodal_encoder():
    # image shape: (batch_size, 3, 256, 256)
    image_sample = torch.randn(32, 3, 256, 256)
    text_sample = 'This is a sample text to test the MultimodalEncoder module.'
    ts_sample = torch.randn(32, 32, 500, 20)

    model = MultimodalEncoder(ts_num_features=20)

    print(model)

    output = model(ts_sample, image_sample, text_sample)
    print("MultimodalEncoder output shape:", str(output.shape))
    return
