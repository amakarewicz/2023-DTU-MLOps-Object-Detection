#TestData
import torch
import pytest
import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project


@pytest.mark.skipif(
    not not os.path.exists(_PROJECT_ROOT + "/data"),
    reason="Data files not found",
)

def test_data_loading_output_is_tensor():
    dataset = x #Data
    assert torch.is_tensor(dataset[0].x), "Nodes are not tensor"
    assert torch.is_tensor(dataset[0].edge_index), "Edges are not tensor"
    assert torch.is_tensor(dataset[0].y), "Classes is not tensor"
    assert torch.is_tensor(dataset[0].train_mask), "Train masks are not tensor"
    assert torch.is_tensor(dataset[0].val_mask), "Val masks are not tensor"
    assert torch.is_tensor(dataset[0].test_mask), "Test masks are not tensor"
    
if __name__ == "__main__":
    test_data_loading_output_is_tensor()
 
