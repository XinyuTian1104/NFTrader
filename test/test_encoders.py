import importlib.util
import torch

def import_encoder(name):
    path = '/Users/crinstaniev/Courses/STATS402/model/encoders/' + name + '_encoder.py'
    # Create a module specification object
    spec = importlib.util.spec_from_file_location(f'{name}_encoder', path)
    # Load the module
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    return encoder


def test_image_encoder():
    test_vit = import_encoder('image').test_vit
    test_vit()
    return

def test_text_encoder():
    test_bert = import_encoder('text').test_bert
    test_bert()
    return

def test_ts_encoder():
    tester = import_encoder('ts').test_ts_encoder
    tester()
    return