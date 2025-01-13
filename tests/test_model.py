from src.my_project.model import model
import torch
import pytest

def test_model():
    """Test the model."""
    dummy_input = torch.randn(1, 1, 28, 28)
    assert model(dummy_input).shape == (1, 10), "Model output shape is incorrect, should be (1, 10)"

def test_input_size():
    """Test the input size"""
    dummy_input = torch.randn(1, 1, 28, 28)
    assert model(dummy_input).shape == (1, 10), "Model output shape is incorrect, should be (1, 10)"

@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_different_batch_sizes(batch_size):
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (batch_size, 10), f"Model output shape is incorrect, should be ({batch_size}, 10)"