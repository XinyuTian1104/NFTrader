from core.model.encoders.multimodal_encoder import test_multimodal_encoder
import pytest
import importlib.util
import os
import shutil
from pathlib import Path


def import_encoder(name):
    path = '/Users/crinstaniev/Courses/STATS402/model/encoders/' + name + '_encoder.py'
    # Create a module specification object
    spec = importlib.util.spec_from_file_location(f'{name}_encoder', path)
    # Load the module
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    return encoder


@pytest.mark.skip(reason="tested manually")
def test_image_encoder():
    test_vit = import_encoder('image').test_vit
    test_vit()
    return


@pytest.mark.skip(reason="tested manually")
def test_text_encoder():
    test_bert = import_encoder('text').test_bert
    test_bert()
    return


@pytest.mark.skip(reason="tested manually")
def test_ts_encoder():
    tester = import_encoder('ts').test_ts_encoder
    tester()
    return


def _test_multimodal_encoder():
    test_multimodal_encoder()
    return
