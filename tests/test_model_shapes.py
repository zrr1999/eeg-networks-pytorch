from __future__ import annotations

import pytest
import torch

from eeg_networks import EEGInception, EEGSpatialLSTM


@pytest.mark.parametrize(
    ("model", "input_tensor", "expected_shape"),
    [
        pytest.param(
            EEGInception(num_classes=2),
            torch.zeros(4, 1, 128, 8),
            (4, 2),
            id="eeg-inception",
        ),
        pytest.param(
            EEGSpatialLSTM(num_classes=2, input_size=16, hidden_size=16),
            torch.zeros(4, 1, 128, 8),
            (4, 2),
            id="eeg-spatial-lstm-shared",
        ),
        pytest.param(
            EEGSpatialLSTM(
                num_classes=2,
                input_size=16,
                hidden_size=16,
                dependent=True,
            ),
            torch.zeros(4, 1, 128, 8),
            (4, 2),
            id="eeg-spatial-lstm-dependent",
        ),
    ],
)
def test_model_output_shapes(
    model: torch.nn.Module, input_tensor: torch.Tensor, expected_shape: tuple[int, int]
) -> None:
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == expected_shape
