import torch
from torch.fx import symbolic_trace
from torch.utils._pytree import tree_map

def get_submodel_up_to_node(model, target_node_name):
    """
    Extract a submodel from the given PyTorch model up to a specific node.

    Args:
        model (torch.nn.Module): The PyTorch model.
        target_node_name (str): The name of the node up to which the submodel should be extracted.

    Returns:
        torch.fx.GraphModule: A new model that computes up to the specified node.
    """
    # Trace the model to get a symbolic graph
    traced = symbolic_trace(model)

    # Create a new graph
    new_graph = torch.fx.Graph()

    # Map old nodes to new nodes
    node_mapping = {}

    # Iterate through the nodes in the original graph
    target_node = None
    for node in traced.graph.nodes:
        # Copy the node to the new graph
        new_node = new_graph.node_copy(node, lambda n: node_mapping[n])
        node_mapping[node] = new_node

        # Stop copying nodes once the target node is reached
        if node.name == target_node_name:
            target_node = node
            break
    
    if target_node is None:
        raise ValueError(f"Node '{target_node_name}' not found in the model. Available nodes are:\n{'\n'.join([node.name for node in traced.graph.nodes])}")

    # Set the output of the new graph to the target node
    new_graph.output(node_mapping[target_node])

    # Create a new GraphModule with the modified graph
    submodel = torch.fx.GraphModule(traced, new_graph)

    return submodel


def get_graph_node_shapes(model, input_shape):
    """
    Get graph node names and tensor shapes for a given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        input_shape (tuple): The shape of the input tensor.

    Returns:
        list: A list of tuples where each tuple contains the graph node name
              and the shape of the tensor returned by it.
    """
    # Create a dummy input tensor with the given shape
    dummy_input = torch.zeros(*input_shape)

    # Trace the model to get a symbolic graph
    traced = symbolic_trace(model)

    # Dictionary to store the shapes of intermediate tensors
    node_shapes = {}

    # Hook to capture the output shapes of each node
    def capture_shapes(module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            node_shapes[module.name] = tuple(outputs.shape)
        else:
            # Handle cases where outputs are tuples or lists of tensors
            node_shapes[module.name] = tree_map(lambda x: tuple(x.shape) if isinstance(x, torch.Tensor) else None, outputs)

    # Register hooks on each node in the graph
    for node in traced.graph.nodes:
        if node.op == 'call_module':
            submodule = dict(traced.named_modules())[node.target]
            submodule.name = node.name  # Assign the node name to the submodule
            submodule.register_forward_hook(capture_shapes)

    # Run the traced model with the dummy input to populate shapes
    traced(dummy_input)

    # Get graph node names
    graph_node_names = [node.name for node in traced.graph.nodes]

    # Create the result list with node names and their corresponding shapes
    result = [(name, node_shapes.get(name, None)) for name in graph_node_names]

    return result


from cabrnet.archs.generic.model import CaBRNet
from torchvision.models import get_model

# Let's use this framework to trace the size of results in the explainable model's backbone extractor
if __name__ == "__main__":

    model_name = "vgg19"
    node_name = "flatten"

    #model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    #print(model)
    model = get_model(model_name)
    model = get_submodel_up_to_node(model, node_name)

    # Input tensor shape
    input_shape = (1,3,256,256) # BCTHW

    # Get graph node names and shapes
    node_shapes = get_graph_node_shapes(model, input_shape)

    # Print the result
    print(f"Intermediate output shapes for model '{model_name}' up to node {node_name}:")
    for node_name, shape in node_shapes:
        if shape is not None:
            print(f"{node_name}\t{shape}")