#TestData

import pytest

@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA + "/processed")
    or not os.path.exists(_PROJECT_ROOT + "/config"),
    reason="Data and config files not found",
)
def test_function():
    ...
