import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


@pytest.mark.skip()
def test_image_encoder():
    from core.model.encoders.image_encoder import test_vit
    test_vit()
    return


@pytest.mark.skip()
def test_text_encoder():
    from core.model.encoders.text_encoder import test_bert
    test_bert()
    return


@pytest.mark.skip()
def test_ts_encoder():
    from core.model.encoders.ts_encoder import test_ts_encoder
    test_ts_encoder()
    return


@pytest.mark.skip()
def test_multimodal_encoder():
    from core.model.encoders.multimodal_encoder import test_multimodal_encoder
    test_multimodal_encoder()
    return
