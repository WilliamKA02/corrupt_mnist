import pytest
import os.path
from torch.utils.data import Dataset
from src.my_project.data import corrupt_mnist

train_set, test_set = corrupt_mnist()

@pytest.mark.skipif(not os.path.exists("data/raw/data/corruptedmnist"), reason="Data files not found")

def test_my_trainset():
    """Test the MyDataset class."""
    assert isinstance(train_set, Dataset)

def test_my_testset():
    """Test the MyDataset class."""
    assert isinstance(test_set, Dataset)

def test_my_trainset_length():
    """Test the MyDataset class."""
    assert len(train_set) == 30000

def test_data():
    """Test the MyDataset class."""
    labels_train = []
    labels_test = []
    for image, label in train_set:
        labels_train.append(int(label))
        assert image.shape == (1, 28, 28), "Image shape is incorrect, should be (1, 28, 28)"
    for image, label in test_set:
        labels_test.append(int(label))
        assert image.shape == (1, 28, 28), "Image shape is incorrect, should be (1, 28, 28)"
    assert len(set(labels_train)) == 10 and len(set(labels_test)) == 10, f"Not all labels are present in the dataset: expected 10, got ({len(set(labels_train)), len(set(labels_test))})"
