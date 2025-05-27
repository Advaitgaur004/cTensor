#!/usr/bin/env python3
"""
Generate reference data using PyTorch for cTensor operator testing.
This script creates hardcoded test data that ensures determinism between
cTensor and PyTorch implementations.
"""

import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def tensor_to_list(tensor: torch.Tensor) -> List[float]:
    """
    Converts a PyTorch tensor to a flat list of floats.
    
    Detaches the tensor from any computation graph, moves it to CPU, flattens it, and returns its elements as a standard Python list of floats.
    """
    return tensor.detach().cpu().flatten().tolist()

def generate_test_tensors() -> Dict[str, Any]:
    """
    Creates a dictionary of predefined test tensors with fixed shapes and values for use in reference data generation.
    
    Returns:
        A dictionary mapping tensor names to their shape and data, including small 1D and 2D tensors, matrix multiplication pairs, batch tensors, softmax logits, and one-hot encoded labels.
    """
    test_data = {}
    
    # Small tensors for basic operations
    test_data['tensor_1d_small'] = {
        'shape': [4],
        'data': [1.0, 2.0, 3.0, 4.0]
    }
    
    test_data['tensor_1d_small_2'] = {
        'shape': [4], 
        'data': [0.5, 1.5, 2.5, 3.5]
    }
    
    test_data['tensor_2d_small'] = {
        'shape': [2, 3],
        'data': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    }
    
    test_data['tensor_2d_small_2'] = {
        'shape': [2, 3],
        'data': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
    
    # Matrix multiplication compatible tensors
    test_data['matrix_a'] = {
        'shape': [2, 3],
        'data': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    }
    
    test_data['matrix_b'] = {
        'shape': [3, 2],
        'data': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    }
    
    # Larger tensors for neural network operations
    test_data['tensor_batch'] = {
        'shape': [2, 4],
        'data': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    }
    
    # Softmax input (logits)
    test_data['logits'] = {
        'shape': [2, 3],
        'data': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    }
    
    # One-hot encoded labels
    test_data['labels_onehot'] = {
        'shape': [2, 3],
        'data': [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    }
    
    return test_data

def generate_arithmetic_operations() -> Dict[str, Any]:
    """
    Generates reference data for element-wise arithmetic operations on fixed 1D tensors.
    
    Returns:
        A dictionary containing inputs, outputs, and shapes for addition, subtraction,
        multiplication, division, power, and scalar operations between two predefined
        1D tensors. All tensor data is converted to lists for serialization.
    """
    results = {}
    
    # Create test tensors
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([0.5, 1.5, 2.5, 3.5])
    
    # Addition
    results['add'] = {
        'input_a': tensor_to_list(a),
        'input_b': tensor_to_list(b),
        'output': tensor_to_list(a + b),
        'shape': list(a.shape)
    }
    
    # Subtraction
    results['sub'] = {
        'input_a': tensor_to_list(a),
        'input_b': tensor_to_list(b),
        'output': tensor_to_list(a - b),
        'shape': list(a.shape)
    }
    
    # Multiplication
    results['mul'] = {
        'input_a': tensor_to_list(a),
        'input_b': tensor_to_list(b),
        'output': tensor_to_list(a * b),
        'shape': list(a.shape)
    }
    
    # Division
    results['div'] = {
        'input_a': tensor_to_list(a),
        'input_b': tensor_to_list(b),
        'output': tensor_to_list(a / b),
        'shape': list(a.shape)
    }
    
    # Power
    results['pow'] = {
        'input_a': tensor_to_list(a),
        'input_b': tensor_to_list(torch.tensor([2.0, 2.0, 2.0, 2.0])),
        'output': tensor_to_list(torch.pow(a, 2.0)),
        'shape': list(a.shape)
    }
    
    # Scalar operations
    scalar = 2.5
    results['add_scalar'] = {
        'input': tensor_to_list(a),
        'scalar': scalar,
        'output': tensor_to_list(a + scalar),
        'shape': list(a.shape)
    }
    
    results['mul_scalar'] = {
        'input': tensor_to_list(a),
        'scalar': scalar,
        'output': tensor_to_list(a * scalar),
        'shape': list(a.shape)
    }
    
    return results

def generate_matrix_operations() -> Dict[str, Any]:
    """
    Generates reference data for matrix operations including matrix multiplication and transpose.
    
    Returns:
        A dictionary containing input tensors, output results, and shape metadata for matrix multiplication and transpose operations, with all tensors converted to lists for serialization.
    """
    results = {}
    
    # Matrix multiplication
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
    
    matmul_result = torch.matmul(a, b)
    
    results['matmul'] = {
        'input_a': tensor_to_list(a),
        'input_b': tensor_to_list(b),
        'output': tensor_to_list(matmul_result),
        'shape_a': list(a.shape),
        'shape_b': list(b.shape),
        'shape_output': list(matmul_result.shape)
    }
    
    # Transpose
    results['transpose'] = {
        'input': tensor_to_list(a),
        'output': tensor_to_list(a.T),
        'shape_input': list(a.shape),
        'shape_output': list(a.T.shape)
    }
    
    return results

def generate_reduction_operations() -> Dict[str, Any]:
    """
    Generates reference data for reduction operations on a fixed 2D tensor.
    
    Returns:
        A dictionary containing inputs, outputs, and shape metadata for sum, mean, max, min, and argmax (along the last dimension) operations.
    """
    results = {}
    
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Sum
    results['sum'] = {
        'input': tensor_to_list(a),
        'output': [torch.sum(a).item()],
        'shape_input': list(a.shape),
        'shape_output': [1]
    }
    
    # Mean
    results['mean'] = {
        'input': tensor_to_list(a),
        'output': [torch.mean(a).item()],
        'shape_input': list(a.shape),
        'shape_output': [1]
    }
    
    # Max
    results['max'] = {
        'input': tensor_to_list(a),
        'output': [torch.max(a).item()],
        'shape_input': list(a.shape),
        'shape_output': [1]
    }
    
    # Min
    results['min'] = {
        'input': tensor_to_list(a),
        'output': [torch.min(a).item()],
        'shape_input': list(a.shape),
        'shape_output': [1]
    }
    
    # Argmax (along last dimension)
    argmax_result = torch.argmax(a, dim=-1)
    results['argmax'] = {
        'input': tensor_to_list(a),
        'output': argmax_result.tolist(),
        'shape_input': list(a.shape),
        'shape_output': list(argmax_result.shape)
    }
    
    return results

def generate_activation_functions() -> Dict[str, Any]:
    """
    Generates reference data for activation and mathematical functions using fixed input tensors.
    
    Returns:
        A dictionary containing inputs, outputs, and shapes for various activation
        (ReLU, sigmoid, tanh, softmax) and mathematical (log, exp, sin, cos, tan)
        functions, with all tensor data converted to lists for serialization.
    """
    results = {}
    
    # Test data with both positive and negative values
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    
    # ReLU
    results['relu'] = {
        'input': tensor_to_list(x),
        'output': tensor_to_list(torch.relu(x)),
        'shape': list(x.shape)
    }
    
    # Sigmoid
    results['sigmoid'] = {
        'input': tensor_to_list(x),
        'output': tensor_to_list(torch.sigmoid(x)),
        'shape': list(x.shape)
    }
    
    # Tanh
    results['tanh'] = {
        'input': tensor_to_list(x),
        'output': tensor_to_list(torch.tanh(x)),
        'shape': list(x.shape)
    }
    
    # Softmax (2D input for batch processing)
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    results['softmax'] = {
        'input': tensor_to_list(logits),
        'output': tensor_to_list(torch.softmax(logits, dim=-1)),
        'shape': list(logits.shape)
    }
    
    # Mathematical functions
    x_pos = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0])  # Positive values for log
    
    results['log'] = {
        'input': tensor_to_list(x_pos),
        'output': tensor_to_list(torch.log(x_pos)),
        'shape': list(x_pos.shape)
    }
    
    results['exp'] = {
        'input': tensor_to_list(x),
        'output': tensor_to_list(torch.exp(x)),
        'shape': list(x.shape)
    }
    
    # Trigonometric functions
    x_trig = torch.tensor([0.0, 0.5, 1.0, 1.5, 3.14159/2, 3.14159])
    
    results['sin'] = {
        'input': tensor_to_list(x_trig),
        'output': tensor_to_list(torch.sin(x_trig)),
        'shape': list(x_trig.shape)
    }
    
    results['cos'] = {
        'input': tensor_to_list(x_trig),
        'output': tensor_to_list(torch.cos(x_trig)),
        'shape': list(x_trig.shape)
    }
    
    results['tan'] = {
        'input': tensor_to_list(x_trig),
        'output': tensor_to_list(torch.tan(x_trig)),
        'shape': list(x_trig.shape)
    }
    
    return results

def generate_loss_functions() -> Dict[str, Any]:
    """
    Generates reference data for loss functions, including cross-entropy and softmax cross-entropy.
    
    Returns:
        A dictionary containing input tensors, computed loss values, and shape metadata for cross-entropy loss (using one-hot labels and predicted probabilities) and softmax cross-entropy loss (using logits and one-hot labels).
    """
    results = {}
    
    # Cross-entropy loss
    y_true = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # One-hot encoded
    y_pred = torch.tensor([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])  # Predicted probabilities
    
    # PyTorch cross-entropy expects log probabilities, but our implementation uses probabilities
    # So we'll compute it manually to match our implementation
    epsilon = 1e-8
    ce_loss = -torch.sum(y_true * torch.log(y_pred + epsilon)) / y_true.shape[0]
    
    results['crossentropy'] = {
        'y_true': tensor_to_list(y_true),
        'y_pred': tensor_to_list(y_pred),
        'output': [ce_loss.item()],
        'shape_true': list(y_true.shape),
        'shape_pred': list(y_pred.shape),
        'shape_output': [1]
    }
    
    # Softmax + Cross-entropy (combined)
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_true_softmax = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    
    # Compute softmax then cross-entropy
    softmax_output = torch.softmax(logits, dim=-1)
    softmax_ce_loss = -torch.sum(y_true_softmax * torch.log(softmax_output + epsilon)) / y_true_softmax.shape[0]
    
    results['softmax_crossentropy'] = {
        'y_true': tensor_to_list(y_true_softmax),
        'logits': tensor_to_list(logits),
        'output': [softmax_ce_loss.item()],
        'shape_true': list(y_true_softmax.shape),
        'shape_logits': list(logits.shape),
        'shape_output': [1]
    }
    
    return results

def generate_neural_network_operations() -> Dict[str, Any]:
    """
    Generates reference data for a neural network linear (fully connected) layer.
    
    Returns:
        A dictionary containing input, weight, bias, and output tensors (as lists), along with their shapes, for a fixed linear layer computation.
    """
    results = {}
    
    # Linear layer (fully connected)
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 3x2
    bias = torch.tensor([0.1, 0.2])  # 2
    
    linear_output = torch.matmul(input_tensor, weight) + bias
    
    results['linear'] = {
        'input': tensor_to_list(input_tensor),
        'weight': tensor_to_list(weight),
        'bias': tensor_to_list(bias),
        'output': tensor_to_list(linear_output),
        'shape_input': list(input_tensor.shape),
        'shape_weight': list(weight.shape),
        'shape_bias': list(bias.shape),
        'shape_output': list(linear_output.shape)
    }
    
    return results

def main():
    """
    Generates all categories of deterministic reference data using PyTorch and saves them as JSON files for use in cTensor operator validation.
    
    Creates a dedicated output directory, compiles reference data for various tensor operations, and writes both a combined JSON file and separate files for each data category.
    """
    print("Generating reference data using PyTorch...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'reference_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all test data
    test_data = {
        'test_tensors': generate_test_tensors(),
        'arithmetic_ops': generate_arithmetic_operations(),
        'matrix_ops': generate_matrix_operations(),
        'reduction_ops': generate_reduction_operations(),
        'activation_functions': generate_activation_functions(),
        'loss_functions': generate_loss_functions(),
        'neural_network_ops': generate_neural_network_operations()
    }
    
    # Save to JSON file
    output_file = os.path.join(output_dir, 'pytorch_reference.json')
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Reference data saved to: {output_file}")
    
    # Also create individual files for easier C integration
    for category, data in test_data.items():
        category_file = os.path.join(output_dir, f'{category}.json')
        with open(category_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Category data saved to: {category_file}")
    
    print("Reference data generation complete!")

if __name__ == '__main__':
    main()