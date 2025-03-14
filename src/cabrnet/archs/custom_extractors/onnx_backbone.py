from __future__ import annotations
from pathlib import PurePath
from typing import Callable, Dict, Optional, Tuple, Union

from loguru import logger
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as rt
import torch
import torch.nn as nn
from torchvision.models._api import register_model

__all__ = ["generic_onnx_model", "GenericONNXModel"]


class ONNXVariantsHandler:
    r"""A class handling multiple variations of a base ONNX file.

    Attributes:
        original_path (PurePath): the original path of provided ONNX file. Not
        modified during inference.

        variants (dict[str, onnx.ModelProto]): A registry
        in charge of storing the variants of ONNX generated by this class.
        The key is expected to be a pipeline name, but can be an arbitrary
        string. No string sanitization is done on the variant name.

    """

    def __init__(self, original_path: PurePath):
        r"""Stores the original path of the model.

        Args:
            original_path (PurePath): Original path of the ONNX model.

        """
        self.original_name = "original"
        self.original_path = original_path
        self.variants: Dict[str, onnx.ModelProto] = {}

    def register_and_save_variant(self, name: str, model: onnx.ModelProto):
        r"""Checks whether the variant does not already exists.
        If not, register
        the variant into the variants registry and save the model under a
        normalized path.

        Args:
            name (str): The variant name.
            model (ModelProto): The model to save.

        """
        if name not in self.variants:
            self.variants[name] = model
            savepath = PurePath("_".join([PurePath(self.original_path).stem, name, ".onnx"]))
            onnx.save_model(model, savepath)
            logger.info(f"Saved result of morphed ONNX at {savepath}")
        else:
            logger.warning(f"Error, variant {name} already registered, doing nothing")
            pass

    def get_variant_path(self, name: str) -> PurePath:
        r"""Returns the path of the ONNX model given its variant name.
        Raises an FileNotFound exception if the variant does not exist in the
        registry.

        Args:
            name (str): The name of the variant.
        """
        if name in self.variants:
            return PurePath("_".join([PurePath(self.original_path).stem, name, ".onnx"]))
        else:
            logger.error(
                f"Warning, ONNX variant {name} not registered in ONNX handler. Trying to load non-existing file. Aborting "
            )
            raise FileNotFoundError

    def get_variant_model(self, name: str) -> Optional[onnx.ModelProto]:
        r"""Returns the ONNX ModelProto corresponding to a variant name.
        Raises an FileNotFound exception if the variant does not exist in the
        registry.

        Args:
            name (str): The name of the variant.
        """
        if name in self.variants:
            return self.variants[name]
        else:
            logger.error(
                f"Warning, ONNX variant {name} not registered in ONNX handler. Trying to load non-existing model. Aborting "
            )
            raise FileNotFoundError

    def safe_onnx_compute(self, f: Callable[..., onnx.ModelProto], variant_name: str, **kwargs):
        r"""Computes a function f on its argument, and save the result of the
        computation in the variants registry.

        Args:
            f (Callable): Function performing a transformation of an ONNX
            model. Assumed to return an ONNX model.

            variant_name (str): Variant name to save.

            kwargs: Arguments of f.
        """

        model = f(**kwargs)
        self.register_and_save_variant(name=variant_name, model=model)

    def __iter__(self):
        r"""Returns an iterator on the key and values of the
        registry."""
        self.iter = iter(self.variants.items())
        return self.iter

    def get_only_modified_variants(self):
        r"""Returns a variant registry without the original ONNX path.
        Filtering out the original backbone is necessary when computing
        batched inputs.
        """
        d = {k: v for k, v in self.variants.items() if k != self.original_name}
        return d


class GenericONNXModel(nn.Module):
    r"""A class describing generic ONNX models to be used as backbone.

    This class provides a forward method to use ONNX models inside CaBRNet,
    relying on the onnxruntime. It does not support training.

    If provided with a dictionary of layers, the class handles the
    generation of alternate ONNX models describing
    the original model, trimed upto a specific given layer.
    This mirrors the original ConvExtractor construction.

    Some assumptions about the loaded ONNX model:
        * it does not have any cycles;
        * it does not have any control structures like conditional or loops;
        * its input and output shapes first dimensions are symbolic.


    Attributes:
        variants: A ONNXVariantsHandler object.
    """

    def __init__(self, onnx_path: str):
        r"""Instanciates a forward-capable class from an ONNX path.

        Args:
            onnx_path (str): The path of the ONNX model to wrap in the class.


        """
        super(GenericONNXModel, self).__init__()
        onnx_purepath = PurePath(onnx_path)
        self.variants = ONNXVariantsHandler(original_path=onnx_purepath)
        model = onnx.load(onnx_purepath)
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model)
        logger.info(f"Loaded ONNX model located at {onnx_purepath} and performed sanity checks on it.")
        self.variants.register_and_save_variant(self.variants.original_name, model)

    def get_original_onnx_model_path(self) -> PurePath:
        r"""Returns the original ONNX model path."""
        return self.variants.original_path

    def get_output_shape_of_node(self, node: onnx.ValueInfoProto) -> Tuple[Union[int, str], int, int, int]:
        r"""Returns the output shape of a given ONNX node.

        Args:
            node (ValueInfoProto): An ONNX node to recover the node from.
        """

        dims = node.type.tensor_type.shape.dim

        shape = [ x.dim_value for x in dims[1:] ]
        if hasattr(dims[0], "dim_param"):
            b_v = dims[0].dim_param
        else:
            b_v = dims[0].dim_value

        shape.insert(0, b_v)
        return tuple(shape)

    def get_output_shape_of_layer(
        self, model: onnx.ModelProto, layer_cut: str
    ) -> Tuple[Union[int, str], int, int, int]:
        r"""Returns the output shape of a layer in a given model.
        The model is expected to have a non-empty graph.value_info field
        (automatically filled when a onnx shape inference is called on it).

        Args:
            model (ModelProto): An ONNX model to get the output shape of.

            layer_cut (str): The name of the layer. The layer name must be
            provided by the user; it is expected to be a valid node name inside
            the ONNX model.
        """
        assert len(model.graph.value_info) > 0
        node_candidates = [x for x in model.graph.value_info if x.name.__contains__(layer_cut)]
        # otherwise, layer_cut is underspecified
        assert len(node_candidates) == 1
        return self.get_output_shape_of_node(node_candidates[0])

    def _trim_model(self, model: onnx.ModelProto, layer_cut: str) -> onnx.ModelProto:
        r"""Trims the original ONNX graph upto the provided layer name
        (included), and saves a new corresponding ONNX graph on disk.
        The original model is preserved.

        Args:
            model (ModelProto): ONNX model to trim.
            layer_cut (str): Layer to trim the model to.

        Returns:
            A modified ONNX model.
        """
        logger.info(f"Performing ONNX model edition to provide an alternate output.")
        to_trim = False
        out_shape = self.get_output_shape_of_layer(model=model, layer_cut=layer_cut)

        graph = gs.import_onnx(model)
        n_to_trim = []
        for node in graph.nodes:
            if node.name == layer_cut:
                to_trim = True
                last_node = node
            if to_trim:
                n_to_trim.append(node)
        for node in n_to_trim:
            last_node.outputs = node.outputs
            node.outputs.clear()
        # create a new output for the graph
        # and remove the previous one
        v = gs.Variable("features", dtype=np.float32, shape=out_shape)
        graph.outputs = [v]
        last_node.outputs = graph.outputs
        logger.debug(f"Replacing output of graph with new node {v} of shape {v.shape}")
        graph.cleanup()
        model_proto = gs.export_onnx(graph)
        return model_proto

    def trim_model(self, return_nodes: dict[str, str]):
        r"""Performs safe computation of model trimming.

        Args:
            return_nodes (dict[str,str]): Should be a
            dict['source_layer']='pipeline_name'.
        """
        for layer_cut, pipeline_name in return_nodes.items():
            model = self.variants.get_variant_model("original")
            self.variants.safe_onnx_compute(
                f=self._trim_model,
                variant_name=pipeline_name,
                **{
                    "model": model,
                    "layer_cut": layer_cut,
                },
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""Performs a forward-pass on all registered model variants.

        The forward pass is done using the ONNX runtime. This comes with several
        limitations:
            * the ONNX model must be present on disk, thus it is loaded for each
            batch;
            * the input and output type is currently limited to float32
            (meaning a model with integer output scores will not work).

        If one of the variant fails to perform its forward pass, the forward
        pass of the whole model fails.

        Note that the inference is never done on the original model, but only on
        preprocessed models generated by the class. It is thus sufficient to
        save the original model and the corresponding inputs arguments to
        regenerate the ONNX variants.

            Args:
                x (Tensor): The tensor to compute the forward pass.

            Returns:
                A dictionary of (variant_name (str), output_tensor (Tensor)).
        """
        device = x.device
        # Only supports CPU execution
        providers = ["CPUExecutionProvider"]
        if device == torch.device("cpu"):
            device_type = "cpu"
        else:
            logger.error(
                f"Error, unable to perform a forward pass on a ONNX backbone with device {torch.device}. Expects cpu. "
            )
            raise NotImplemented
        ort_sessions = {}
        outputs = {}
        batch_size = x.size()[0]
        for variant in self.variants.get_only_modified_variants().keys():
            path = self.variants.get_variant_path(variant)
            logger.debug(f"Performing ONNX inference of {variant} located at {path}")
            ort_sessions[variant] = rt.InferenceSession(path, providers=providers)
            input_of_sess = ort_sessions[variant].get_inputs()[0]
            output_of_sess = ort_sessions[variant].get_outputs()[0]
            input_name = input_of_sess.name
            output_name = output_of_sess.name
            bind = ort_sessions[variant].io_binding()
            match output_of_sess.shape:
                case (_, d):
                    out_shape = (batch_size, d)
                case (_, h, w, d):
                    out_shape = (batch_size, h, w, d)
                case _:
                    assert False

            out_tensor = torch.empty(
                size=torch.Size(out_shape),
                dtype=torch.float32,
                device=device,
            ).contiguous()

            bind.bind_input(
                name=input_name,
                device_type=device_type,
                device_id=0,
                element_type=np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            bind.bind_output(
                name=output_name,
                device_type=device_type,
                device_id=0,
                element_type=np.float32,
                shape=tuple(out_tensor.shape),
                buffer_ptr=out_tensor.data_ptr(),
            )
            ort_sessions[variant].run_with_iobinding(bind)
            outputs[variant] = out_tensor
        return outputs


@register_model()
def generic_onnx_model(onnx_path: str) -> GenericONNXModel:
    r"""Registers the GenericONNXModel to PyTorch models APIs.

    Args:
        onnx_path (str): The path of the ONNX model to wrap in the class.
    """
    return GenericONNXModel(onnx_path=onnx_path)
