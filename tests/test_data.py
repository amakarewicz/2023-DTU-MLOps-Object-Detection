#TestData

import pytest
import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project


@pytest.mark.skipif(
    not not os.path.exists(_PROJECT_ROOT + "/data"),
    reason="Data files not found",
)
def test_function():
    ...
