from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Generator


# make all tests run in torch.no_grad() context
@pytest.fixture(autouse=True)
def torch_no_grad() -> Generator[None]:
    with torch.no_grad():
        yield
