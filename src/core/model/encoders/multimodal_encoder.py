import torch
import torch.nn as nn

from core.model.encoders.image_encoder import ViTEncoder
from core.model.encoders.text_encoder import BertEncoder
from core.model.encoders.ts_encoder import TimeSeriesEncoder


class MultimodalEncoder(nn.Module):
    def __init__(self, ts_num_features, img_image_size=224, img_patch_size=16, img_emb_size=256, img_num_heads=8, img_num_layers=12, img_feature_dim=2048, img_dropout=0.1, img_channel=3, text_hidden_size=768, text_output_size=2048, ts_num_hidden=256, ts_num_layers=6, ts_num_heads=8, ts_output_size=2048):
        super(MultimodalEncoder, self).__init__()

        self.image_encoder = ViTEncoder(
            image_size=img_image_size,
            patch_size=img_patch_size,
            emb_size=img_emb_size,
            num_heads=img_num_heads,
            num_layers=img_num_layers,
            feature_dim=img_feature_dim,
            dropout=img_dropout,
            channel=img_channel
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
    # (batch_size, channels, height, width)
    image_sample = torch.randn(1, 3, 224, 224)
    text_sample = [
        'This is a sample text to test the MultimodalEncoder module.'
        for _ in range(1)
    ]  # (batch_size, x)
    # (batch_size, timesteps, num_features)
    ts_sample = torch.randn(1, 16, 5)

    model = MultimodalEncoder(ts_num_features=5)

    print(model)

    output = model(ts_sample, image_sample, text_sample)
    print("MultimodalEncoder output shape:", str(output.shape))
    return
