import onnx
import argparse
from pathlib import Path
from onnx import optimizer
from quantizer.quant_utils import generate_identified_filename
from onnxruntime import SessionOptions, InferenceSession, GraphOptimizationLevel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    args = parser.parse_args()
    return args

def optimize_model(model_path: Path):
    '''
        Generate model that applies graph optimization (constant folding,etc.)
        parameter model_path: path to the original onnx model
        return: optimized onnx model
    '''
    opt_model_path = generate_identified_filename(model_path, "-opt")
    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    _ = InferenceSession(model_path.as_posix(), sess_option)
    optimized_model = onnx.load(opt_model_path.as_posix())
    return optimized_model

def replace_gemm_with_matmul(model):
    nodes_to_remove = []
    nodes_to_add = []
    for node in model.graph.node:
        if node.op_type == 'Gemm':
            alpha = 1.0
            beta = 1.0
            transA = 0
            transB = 0
            for attr in node.attribute:
                if attr.name == 'alpha':
                    alpha = onnx.helper.get_attribute_value(attr)
                elif attr.name == 'beta':
                    beta = onnx.helper.get_attribute_value(attr)
                elif attr.name == 'transA':
                    transA = onnx.helper.get_attribute_value(attr)
                elif attr.name == 'transB':
                    transB = onnx.helper.get_attribute_value(attr)
            if alpha == 1.0 and beta == 1.0 and transA == 0 and transB == 0:
                matmul_node = onnx.helper.make_node(
                    'MatMul',
                    [node.input[0], node.input[1]],
                    [node.output[0]+'_MatMul'],
                    name=node.output[0]+'_MatMul')

                add_node = onnx.helper.make_node(
                    'Add',
                    inputs=[node.output[0]+'_MatMul', node.input[2]],
                    outputs=node.output,
                    name=node.output[0]+'_Add')

                nodes_to_remove.extend([node])
                nodes_to_add.extend([matmul_node, add_node])

    model.graph.node.extend(nodes_to_add)
    for node in nodes_to_remove:
        model.graph.node.remove(node)

def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Warning: Model with ir_version below 4 requires initializer in graph input, so not removing'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])


def add_value_info_for_constants(model):
    """
    https://github.com/onnx/onnx/issues/2995
    
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    add_const_value_infos_to_graph(model.graph)


def sum_to_add(model):
    '''
    Convert an add with 2 operators into a sum
    '''
    for node in model.graph.node:
        if node.op_type == 'Sum' and len(node.input) == 2:
            node.op_type = 'Add'

def transforms():
    args = get_args()
    model = onnx.load(args.input)

    # https://github.com/onnx/onnx/issues/2902
    add_value_info_for_constants(model)
    for init in model.graph.initializer:
        for value_info in model.graph.value_info:
            if init.name == value_info.name:
                model.graph.input.append(value_info)

    optimized = optimizer.optimize(model, ['extract_constant_to_initializer', 'fuse_bn_into_conv'])
    remove_initializer_from_input(optimized)
    replace_gemm_with_matmul(optimized)
    sum_to_add(optimized)
    onnx.save(optimized, args.output)

if __name__ == '__main__':
    transforms()