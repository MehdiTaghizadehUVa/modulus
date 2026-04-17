# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
REPO_ROOT = EXAMPLE_ROOT.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing import FloodGINODataProcessor  # noqa: E402
from models import CNNDomainClassifier, GINOWrapper, ImportableTorchModuleAdapter  # noqa: E402
from tests.model_fixtures import FakeGINOBackbone  # noqa: E402


FIXTURE_DIR = Path(__file__).resolve().parent / "data"


def _load_fixture(file_name: str):
    return torch.load(FIXTURE_DIR / file_name, map_location="cpu", weights_only=False)


def _clone_structure(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_structure(item) for key, item in value.items()}
    return deepcopy(value)


@pytest.mark.parametrize(
    ("backbone_kwargs", "wrapper_kwargs", "expected"),
    [
        (
            {},
            {},
            {
                "autoregressive": False,
                "residual_output": False,
                "max_autoregressive_steps": 1,
            },
        ),
        (
            {"fno_hidden_channels": 4, "out_channels": 3, "scale": 0.25},
            {
                "autoregressive": True,
                "residual_output": True,
                "max_autoregressive_steps": 4,
            },
            {
                "autoregressive": True,
                "residual_output": True,
                "max_autoregressive_steps": 4,
            },
        ),
    ],
    ids=["defaults", "custom"],
)
def test_gino_wrapper_constructor_attributes(backbone_kwargs, wrapper_kwargs, expected):
    wrapper = GINOWrapper(FakeGINOBackbone(**backbone_kwargs), **wrapper_kwargs)

    assert isinstance(wrapper.model, ImportableTorchModuleAdapter)
    assert wrapper.autoregressive is expected["autoregressive"]
    assert wrapper.residual_output is expected["residual_output"]
    assert wrapper.max_autoregressive_steps == expected["max_autoregressive_steps"]


@pytest.mark.parametrize(
    ("classifier_kwargs", "expected_fc_dim", "expected_lambda"),
    [
        (
            {
                "in_channels": 4,
                "conv_layers": [],
                "fc_dim": 1,
            },
            1,
            1.0,
        ),
        (
            {
                "in_channels": 4,
                "lambda_max": 0.7,
                "da_cfg": {
                    "conv_layers": [{"out_channels": 5, "kernel_size": 3, "pool_size": 2}],
                    "fc_dim": 2,
                },
            },
            2,
            0.7,
        ),
    ],
    ids=["defaults", "custom"],
)
def test_cnn_domain_classifier_constructor_attributes(
    classifier_kwargs,
    expected_fc_dim,
    expected_lambda,
):
    classifier = CNNDomainClassifier(**classifier_kwargs)

    assert classifier.in_channels == 4
    assert classifier.lambda_max == expected_lambda
    assert classifier.fc.out_features == expected_fc_dim
    assert classifier.grl.lambda_ == expected_lambda


@pytest.mark.parametrize(
    ("processor_kwargs", "expected_inverse_test"),
    [
        ({"device": "cpu"}, True),
        ({"device": "cpu", "inverse_test": False}, False),
    ],
    ids=["defaults", "custom"],
)
def test_flood_gino_data_processor_constructor_attributes(
    processor_kwargs,
    expected_inverse_test,
):
    processor = FloodGINODataProcessor(**processor_kwargs)

    assert processor.model is None
    assert processor.inverse_test is expected_inverse_test
    assert processor._device_str == "cpu"


def test_gino_wrapper_reference_tensor_output():
    fixture = _load_fixture("gino_wrapper_reference.pth")
    wrapper = GINOWrapper(FakeGINOBackbone())

    output = wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=fixture["output_queries"],
        x=fixture["x"],
    )

    torch.testing.assert_close(output, fixture["tensor_output"], rtol=1e-6, atol=1e-6)


def test_gino_wrapper_reference_dict_output():
    fixture = _load_fixture("gino_wrapper_reference.pth")
    wrapper = GINOWrapper(FakeGINOBackbone())

    output = wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=_clone_structure(fixture["dict_output_queries"]),
        x=fixture["x"],
    )

    assert output.keys() == fixture["dict_output"].keys()
    for key in output:
        torch.testing.assert_close(output[key], fixture["dict_output"][key], rtol=1e-6, atol=1e-6)


def test_gino_wrapper_autoregressive_residual_output():
    fixture = _load_fixture("gino_wrapper_reference.pth")
    base_wrapper = GINOWrapper(FakeGINOBackbone())
    autoregressive_wrapper = GINOWrapper(
        FakeGINOBackbone(),
        autoregressive=True,
        residual_output=True,
    )

    base_output = base_wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=fixture["output_queries"],
        x=fixture["x"],
    )
    autoregressive_output = autoregressive_wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=fixture["output_queries"],
        x=fixture["x"],
    )

    expected = base_output + fixture["x"][:, :, -3:]
    torch.testing.assert_close(autoregressive_output, expected, rtol=1e-6, atol=1e-6)


def test_cnn_domain_classifier_reference_output():
    fixture = _load_fixture("domain_classifier_reference.pth")
    classifier = CNNDomainClassifier(
        in_channels=fixture["config"]["in_channels"],
        lambda_max=fixture["config"]["lambda_max"],
        da_cfg=fixture["config"]["da_cfg"],
    )
    classifier.load_state_dict(fixture["state_dict"])

    output = classifier(fixture["input"])

    torch.testing.assert_close(output, fixture["output"], rtol=1e-6, atol=1e-6)


def test_flood_gino_data_processor_reference_behaviour():
    fixture = _load_fixture("processor_reference.pth")
    processor = FloodGINODataProcessor(device="cpu", inverse_test=False)
    processor.eval()

    processed = processor.preprocess(_clone_structure(fixture["sample"]))
    for key, expected in fixture["processed"].items():
        actual = processed.get(key)
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    post_out, post_sample = processor.postprocess(
        fixture["postprocess_input"].clone(),
        _clone_structure(processed),
    )
    torch.testing.assert_close(post_out, fixture["postprocess_output"], rtol=1e-6, atol=1e-6)
    for key in fixture["processed"]:
        if fixture["processed"][key] is not None:
            torch.testing.assert_close(post_sample[key], fixture["processed"][key], rtol=1e-6, atol=1e-6)


def test_gino_wrapper_from_checkpoint_smoke():
    fixture = _load_fixture("gino_wrapper_reference.pth")
    wrapper = GINOWrapper.from_checkpoint(str(FIXTURE_DIR / "gino_wrapper.mdlus"))

    output = wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=fixture["output_queries"],
        x=fixture["x"],
    )

    torch.testing.assert_close(output, fixture["tensor_output"], rtol=1e-6, atol=1e-6)


def test_cnn_domain_classifier_from_checkpoint_roundtrip():
    fixture = _load_fixture("domain_classifier_reference.pth")
    classifier = CNNDomainClassifier.from_checkpoint(str(FIXTURE_DIR / "domain_classifier.mdlus"))

    output = classifier(fixture["input"])

    torch.testing.assert_close(output, fixture["output"], rtol=1e-6, atol=1e-6)


def test_flood_gino_data_processor_from_checkpoint_roundtrip():
    fixture = _load_fixture("processor_reference.pth")
    processor = FloodGINODataProcessor.from_checkpoint(str(FIXTURE_DIR / "processor.mdlus"))

    processed = processor.preprocess(_clone_structure(fixture["sample"]))
    for key, expected in fixture["processed"].items():
        if expected is not None:
            torch.testing.assert_close(processed[key], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("output_queries", "error_type", "message"),
    [
        (["bad"], TypeError, "torch.Tensor or a non-empty dict"),
        ({}, ValueError, "must not be empty"),
        ({"main": "bad"}, TypeError, "must be a torch.Tensor"),
        (torch.zeros(1, 2, 3, 2), ValueError, "rank 2 or 3"),
        (torch.zeros(6, 3), ValueError, "coordinate dimension"),
        (
            {"a": torch.zeros(6, 2), "b": torch.zeros(1, 6, 2)},
            ValueError,
            "same batch style",
        ),
    ],
)
def test_gino_wrapper_rejects_invalid_output_queries(output_queries, error_type, message):
    wrapper = GINOWrapper(FakeGINOBackbone())
    fixture = _load_fixture("gino_wrapper_reference.pth")

    with pytest.raises(error_type, match=message):
        wrapper(
            input_geom=fixture["input_geom"],
            latent_queries=fixture["latent_queries"],
            output_queries=output_queries,
            x=fixture["x"],
        )
