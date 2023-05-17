import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


# @pytest.mark.skip()
def test_decision_network():
    from model.decision_network.network import test_decision_network
    test_decision_network()
