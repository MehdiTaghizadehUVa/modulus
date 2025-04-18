# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: E402
import os
import sys

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common
import pytest
from graphcast.utils import create_random_input, fix_random_seeds
from pytest_utils import import_or_fail

# Disable flash attention, specify minimum te version
os.environ["NVTE_FLASH_ATTN"] = "0"
te_version = "2.0.6"


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_forward(device, pytestconfig, num_channels=2, res_h=10, res_w=20):
    """Test graphcast forward pass"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    model_kwds = {
        "mesh_level": 1,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": num_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": num_channels,
        "processor_layers": 3,
        "hidden_dim": 4,
        "do_concat_trick": True,
    }

    fix_random_seeds()
    x = create_random_input(model_kwds["input_res"], model_kwds["input_dim_grid_nodes"])
    x = x.to(device)

    # Construct graphcast model
    model = GraphCastNet(**model_kwds).to(device)

    assert common.validate_forward_accuracy(model, (x,), rtol=1e-2)


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_constructor(
    device, pytestconfig, num_channels_1=2, num_channels_2=3, res_h=10, res_w=20
):
    """Test graphcast constructor options"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    # Define dictionary of constructor args
    arg_list = [
        {
            "mesh_level": 1,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels_1,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels_1,
            "processor_layers": 3,
            "hidden_dim": 4,
            "do_concat_trick": True,
        },
        {
            "multimesh_level": 1,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels_2,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels_2,
            "processor_layers": 4,
            "hidden_dim": 8,
            "do_concat_trick": False,
        },
    ]
    for kw_args in arg_list:
        # Construct GraphCast model
        model = GraphCastNet(**kw_args).to(device)

        x = create_random_input(
            kw_args["input_res"], kw_args["input_dim_grid_nodes"]
        ).to(device)
        outvar = model(x)
        assert outvar.shape == (
            1,
            kw_args["output_dim_grid_nodes"],
            *kw_args["input_res"],
        )


@import_or_fail(["dgl", "transformer_engine"], [None, te_version])
def test_graphcast_te_constructor(
    pytestconfig, num_channels_1=2, num_channels_2=3, res_h=10, res_w=20
):
    """Test graphcast constructor options with graph transformer processor"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    device = "cuda:0"  # TE requires cuda

    # Define dictionary of constructor args
    arg_list = [
        {
            "mesh_level": 1,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels_1,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels_1,
            "processor_layers": 3,
            "hidden_dim": 4,
            "do_concat_trick": True,
        },
        {
            "multimesh_level": 1,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels_2,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels_2,
            "processor_layers": 4,
            "hidden_dim": 8,
            "do_concat_trick": False,
            "multimesh": False,
            "processor_type": "GraphTransformer",
            "khop_neighbors": 2,
            "num_attention_heads": 2,
        },
    ]
    for kw_args in arg_list:
        # Construct GraphCast model
        model = GraphCastNet(**kw_args).to(device)

        x = create_random_input(
            kw_args["input_res"], kw_args["input_dim_grid_nodes"]
        ).to(device)
        outvar = model(x)
        assert outvar.shape == (
            1,
            kw_args["output_dim_grid_nodes"],
            *kw_args["input_res"],
        )


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_constructor_backward_compatibility(device, pytestconfig):
    """Test graphcast constructor for backward compatibility for multimesh_level -> mesh_level"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    # Define dictionary of constructor args
    kw_args = {
        "input_res": (10, 20),
        "input_dim_grid_nodes": 2,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": 2,
        "processor_layers": 3,
        "hidden_dim": 4,
        "do_concat_trick": True,
    }
    # Construct GraphCast model
    model_1 = GraphCastNet(**kw_args, mesh_level=1).to(device)
    model_2 = GraphCastNet(**kw_args, multimesh_level=1).to(device)
    model_1_params = sum(p.numel() for p in model_1.parameters())
    model_2_params = sum(p.numel() for p in model_2.parameters())
    assert model_1_params == model_2_params


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_GraphCast_optims(device, pytestconfig, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast optimizations"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        model_kwds = {
            "mesh_level": 1,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels,
            "processor_layers": 3,
            "hidden_dim": 2,
            "do_concat_trick": True,
        }
        fix_random_seeds()
        x = create_random_input(
            model_kwds["input_res"], model_kwds["input_dim_grid_nodes"]
        )
        x = x.to(device)

        # Construct GraphCast model
        model = GraphCastNet(**model_kwds).to(device)
        return model, (x,)

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@import_or_fail(["dgl", "transformer_engine"], [None, te_version])
def test_GraphCast_te_optims(pytestconfig, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast optimizations with graph transformer processor"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    device = "cuda:0"

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        model_kwds = {
            "mesh_level": 1,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels,
            "processor_layers": 3,
            "hidden_dim": 2,
            "do_concat_trick": True,
            "multimesh": False,
            "processor_type": "GraphTransformer",
            "khop_neighbors": 2,
            "num_attention_heads": 2,
        }
        fix_random_seeds()
        x = create_random_input(
            model_kwds["input_res"], model_kwds["input_dim_grid_nodes"]
        )
        x = x.to(device)

        # Construct GraphCast model
        model = GraphCastNet(**model_kwds).to(device)
        return model, (x,)

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_checkpoint(device, pytestconfig, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast checkpoint save/load"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    model_kwds = {
        "mesh_level": 1,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": num_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": num_channels,
        "processor_layers": 3,
        "hidden_dim": 2,
        "do_concat_trick": True,
    }

    # Construct GraphCast model
    model_1 = GraphCastNet(**model_kwds).to(device)
    model_2 = GraphCastNet(**model_kwds).to(device)

    x = create_random_input(model_kwds["input_res"], model_kwds["input_dim_grid_nodes"])
    x = x.to(device)

    assert common.validate_checkpoint(
        model_1,
        model_2,
        (x,),
    )


@import_or_fail(["dgl", "transformer_engine"], [None, te_version])
def test_graphcast_checkpoint_te(pytestconfig, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast checkpoint save/load with graph transformer processor"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    device = "cuda:0"

    model_kwds = {
        "mesh_level": 1,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": num_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": num_channels,
        "processor_layers": 3,
        "hidden_dim": 2,
        "do_concat_trick": True,
        "multimesh": False,
        "processor_type": "GraphTransformer",
        "khop_neighbors": 2,
        "num_attention_heads": 2,
    }

    # Construct GraphCast model
    model_1 = GraphCastNet(**model_kwds).to(device)
    model_2 = GraphCastNet(**model_kwds).to(device)

    x = create_random_input(model_kwds["input_res"], model_kwds["input_dim_grid_nodes"])
    x = x.to(device)

    assert common.validate_checkpoint(
        model_1,
        model_2,
        (x,),
    )


@import_or_fail("dgl")
@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_GraphCast_deploy(device, pytestconfig, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast deployment support"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    model_kwds = {
        "mesh_level": 1,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": num_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": num_channels,
        "processor_layers": 3,
        "hidden_dim": 2,
        "do_concat_trick": True,
    }

    # Construct GraphCast model
    model = GraphCastNet(**model_kwds).to(device)

    x = create_random_input(model_kwds["input_res"], model_kwds["input_dim_grid_nodes"])
    x = x.to(device)

    assert common.validate_onnx_export(model, x)
    assert common.validate_onnx_runtime(model, x)


@import_or_fail(["dgl", "transformer_engine"], [None, te_version])
@common.check_ort_version()
def test_GraphCast_deploy_te(pytestconfig, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast deployment support with graph transformer processor"""

    from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet

    device = "cuda:0"

    model_kwds = {
        "mesh_level": 1,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": num_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": num_channels,
        "processor_layers": 3,
        "hidden_dim": 2,
        "do_concat_trick": True,
        "multimesh": False,
        "processor_type": "GraphTransformer",
        "khop_neighbors": 2,
        "num_attention_heads": 2,
    }

    # Construct GraphCast model
    model = GraphCastNet(**model_kwds).to(device)

    x = create_random_input(model_kwds["input_res"], model_kwds["input_dim_grid_nodes"])
    x = x.to(device)

    assert common.validate_onnx_export(model, x)
    assert common.validate_onnx_runtime(model, x)
