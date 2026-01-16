"""
Utility functions and classes for Dynamic Sparse Training (DST) tutorial.

This module contains:
- Network architectures (SimpleNet, MaskedNet)
- Masked layer implementations (MaskedConv2d, MaskedLinear)
- DST algorithms (pruning, regrowth)
- Sparse matrix conversion and operations
- Training and evaluation functions
- Benchmarking utilities
- Visualization functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import copy
import os

# ============================================================================
# CuPy Setup for Truly Sparse Operations
# ============================================================================
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    
    # Check if GPU is available for CuPy
    try:
        test_array = cp.array([1.0])
        CUPY_GPU_AVAILABLE = True
        del test_array
    except:
        CUPY_GPU_AVAILABLE = False
except ImportError:
    import sys
    print("="*60)
    print("ERROR: CuPy is required but not available!")
    print("="*60)
    print("Please install CuPy to run this script:")
    print("  pip install cupy")
    print("\nOr for CPU-only version:")
    print("  pip install cupy-cpu")
    print("="*60)
    sys.exit(1)


# ============================================================================
# Network Architectures
# ============================================================================

class SimpleNet(nn.Module):
    """
    A simple CNN with:
    - 2 Convolutional layers
    - 4 Fully Connected (Dense) layers
    """
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Dense layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  
        self.fc2 = nn.Linear(128, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, num_classes)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ============================================================================
# Masked Layers for Simulated Sparsity
# ============================================================================

class MaskedConv2d(nn.Module):
    """
    Convolutional layer with a binary mask for simulated sparsity.
    The mask zeros out certain weights, but the dense weights are still stored.
    """
    def __init__(self, conv_layer, sparsity=0.5):
        super(MaskedConv2d, self).__init__()
        self.conv = conv_layer
        self.sparsity = sparsity
        
        # Initialize binary mask (1 = keep weight, 0 = prune weight)
        self.register_buffer('mask', torch.ones_like(conv_layer.weight))
        
        # Initialize mask with desired sparsity
        self.initialize_mask()
    
    def initialize_mask(self):
        """Randomly initialize mask with specified sparsity"""
        num_params = self.mask.numel()
        num_prune = int(self.sparsity * num_params)
        
        # Randomly select weights to prune
        indices = torch.randperm(num_params)[:num_prune]
        flat_mask = self.mask.view(-1)
        flat_mask[indices] = 0
        self.mask = flat_mask.view_as(self.conv.weight)
    
    def forward(self, x):
        # Apply mask to weights (simulated sparsity)
        masked_weight = self.conv.weight * self.mask
        return F.conv2d(x, masked_weight, self.conv.bias, 
                        self.conv.stride, self.conv.padding, 
                        self.conv.dilation, self.conv.groups)


class MaskedLinear(nn.Module):
    """
    Linear (Dense) layer with a binary mask for simulated sparsity.
    """
    def __init__(self, linear_layer, sparsity=0.5):
        super(MaskedLinear, self).__init__()
        self.linear = linear_layer
        self.sparsity = sparsity
        
        # Initialize binary mask
        self.register_buffer('mask', torch.ones_like(linear_layer.weight))
        self.initialize_mask()
    
    def initialize_mask(self):
        """Randomly initialize mask with specified sparsity"""
        num_params = self.mask.numel()
        num_prune = int(self.sparsity * num_params)
        
        indices = torch.randperm(num_params)[:num_prune]
        flat_mask = self.mask.view(-1)
        flat_mask[indices] = 0
        self.mask = flat_mask.view_as(self.linear.weight)
    
    def forward(self, x):
        # Apply mask to weights (simulated sparsity)
        masked_weight = self.linear.weight * self.mask
        return F.linear(x, masked_weight, self.linear.bias)


class MaskedNet(nn.Module):
    """
    Network with masked layers for simulated sparsity.
    """
    def __init__(self, base_net, sparsity=0.5):
        super(MaskedNet, self).__init__()
        
        # Replace conv layers with masked versions
        self.masked_conv1 = MaskedConv2d(base_net.conv1, sparsity)
        self.masked_conv2 = MaskedConv2d(base_net.conv2, sparsity)
        
        # Replace dense layers with masked versions
        self.masked_fc1 = MaskedLinear(base_net.fc1, sparsity)
        self.masked_fc2 = MaskedLinear(base_net.fc2, sparsity)
        self.masked_fc3 = MaskedLinear(base_net.fc3, sparsity)
        self.masked_fc4 = MaskedLinear(base_net.fc4, sparsity)
        
        self.pool = base_net.pool
        
    def forward(self, x):
        x = self.pool(F.relu(self.masked_conv1(x)))
        x = self.pool(F.relu(self.masked_conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.masked_fc1(x))
        x = F.relu(self.masked_fc2(x))
        x = F.relu(self.masked_fc3(x))
        x = self.masked_fc4(x)
        return x
    
    def get_masks(self):
        """Return all masks for inspection"""
        return {
            'conv1': self.masked_conv1.mask,
            'conv2': self.masked_conv2.mask,
            'fc1': self.masked_fc1.mask,
            'fc2': self.masked_fc2.mask,
            'fc3': self.masked_fc3.mask,
            'fc4': self.masked_fc4.mask
        }
    
    def get_sparsity_stats(self):
        """Calculate actual sparsity for each layer"""
        stats = {}
        masks = self.get_masks()
        for name, mask in masks.items():
            total = mask.numel()
            zeros = (mask == 0).sum().item()
            sparsity = zeros / total
            stats[name] = {'total': total, 'zeros': zeros, 'sparsity': sparsity}
        return stats
    
    def get_masks_as_2d(self):
        """Extract masks as 2D arrays for visualization"""
        masks_2d = {}
        masks = self.get_masks()
        
        for name, mask in masks.items():
            mask_np = mask.cpu().numpy()
            
            if len(mask_np.shape) == 4:  # Conv layer: [out_channels, in_channels, H, W]
                # For conv layers, show a representative 2D view
                mask_2d = mask_np[0, 0, :, :].copy()
            elif len(mask_np.shape) == 2:  # FC layer: [out_features, in_features]
                # FC layers are already 2D
                mask_2d = mask_np
            else:
                # Flatten to 2D if needed
                size = int(np.sqrt(mask_np.size))
                if size * size == mask_np.size:
                    mask_2d = mask_np.reshape(size, size)
                else:
                    # If not a perfect square, pad or reshape differently
                    mask_2d = mask_np.flatten()[:size*size].reshape(size, size)
            
            masks_2d[name] = mask_2d
        
        return masks_2d


# ============================================================================
# Dynamic Sparse Training (DST) Functions
# ============================================================================

def magnitude_prune(masked_layer, prune_ratio=0.1):
    """
    Prune weights with smallest magnitude.
    
    Args:
        masked_layer: MaskedConv2d or MaskedLinear layer
        prune_ratio: Fraction of remaining active weights to prune
    
    Returns:
        Number of weights pruned
    """
    # Get weight tensor
    weight = masked_layer.conv.weight if isinstance(masked_layer, MaskedConv2d) else masked_layer.linear.weight
    
    # Get current active weights (where mask == 1)
    flat_mask = masked_layer.mask.view(-1)
    active_indices = torch.where(flat_mask == 1)[0]
    
    if len(active_indices) == 0:
        return 0
    
    # Get magnitudes of active weights
    flat_weight = weight.view(-1)
    active_magnitudes = flat_weight[active_indices].abs()
    
    # Calculate threshold based on magnitude
    num_prune = int(prune_ratio * len(active_indices))
    if num_prune == 0:
        return 0
    
    # Get indices of smallest magnitude weights among active ones
    _, local_indices = torch.topk(active_magnitudes, num_prune, largest=False)
    prune_indices = active_indices[local_indices]
    
    # Update mask
    flat_mask[prune_indices] = 0
    masked_layer.mask = flat_mask.view_as(weight)
    
    return num_prune


def random_regrow(masked_layer, num_regrow):
    """
    Randomly regrow pruned connections.
    
    Args:
        masked_layer: MaskedConv2d or MaskedLinear layer
        num_regrow: Number of pruned weights to regrow (to maintain sparsity)
    """
    # Get pruned weights (where mask == 0)
    pruned_mask = masked_layer.mask == 0
    num_pruned = pruned_mask.sum().item()
    
    if num_pruned == 0 or num_regrow == 0:
        return
    
    # Limit regrowth to available pruned weights
    num_regrow = min(num_regrow, num_pruned)
    
    # Get flat indices of pruned weights
    flat_mask = masked_layer.mask.view(-1)
    pruned_indices = torch.where(flat_mask == 0)[0]
    regrow_indices = pruned_indices[torch.randperm(len(pruned_indices))[:num_regrow]]
    
    # Update mask
    flat_mask[regrow_indices] = 1
    masked_layer.mask = flat_mask.view_as(masked_layer.conv.weight if isinstance(masked_layer, MaskedConv2d) else masked_layer.linear.weight)
    
    # Reinitialize regrown weights (common practice in DST)
    if isinstance(masked_layer, MaskedConv2d):
        weight = masked_layer.conv.weight.data.clone()
        flat_weight = weight.view(-1)
        flat_weight[regrow_indices] = torch.randn(len(regrow_indices), device=weight.device, dtype=weight.dtype) * 0.01
        masked_layer.conv.weight.data = flat_weight.view_as(weight)
    else:
        weight = masked_layer.linear.weight.data.clone()
        flat_weight = weight.view(-1)
        flat_weight[regrow_indices] = torch.randn(len(regrow_indices), device=weight.device, dtype=weight.dtype) * 0.01
        masked_layer.linear.weight.data = flat_weight.view_as(weight)


def apply_dst_step(model, prune_ratio=0.1):
    """
    Apply one step of Dynamic Sparse Training:
    1. Prune small magnitude weights
    2. Regrow the same number of random connections (to maintain target sparsity)
    
    Args:
        model: MaskedNet model
        prune_ratio: Fraction of active weights to prune
    """
    # Apply to all masked layers
    for module in model.modules():
        if isinstance(module, (MaskedConv2d, MaskedLinear)):
            # Prune weights and get the number pruned
            num_pruned = magnitude_prune(module, prune_ratio)
            # Regrow the same number to maintain sparsity
            random_regrow(module, num_pruned)


# ============================================================================
# Truly Sparse Implementation (CSR Matrices)
# ============================================================================

def convert_to_csr_sparse(masked_net, use_gpu=False):
    """
    Convert a masked network to truly sparse format using CSR matrices.
    
    Args:
        masked_net: MaskedNet model
        use_gpu: Whether to create CSR matrices on GPU (if available)
        
    Returns:
        Dictionary mapping layer names to CSR matrices and metadata
    """
    sparse_layers = {}
    use_gpu = use_gpu and CUPY_GPU_AVAILABLE
    
    for name, module in masked_net.named_modules():
        if isinstance(module, MaskedLinear):
            # Get masked weight
            weight = module.linear.weight.data.cpu().numpy()
            mask = module.mask.cpu().numpy()
            masked_weight = weight * mask
            
            # Convert to CSR format using CuPy (GPU or CPU)
            masked_weight_cp = cp.asarray(masked_weight)
            sparse_matrix = cp_sparse.csr_matrix(masked_weight_cp)
            
            # Handle bias
            if module.linear.bias is not None:
                bias = cp.asarray(module.linear.bias.data.cpu().numpy())
            else:
                bias = None
            
            # Calculate actual sparsity
            total_elements = masked_weight.shape[0] * masked_weight.shape[1]
            nnz = sparse_matrix.nnz
            actual_sparsity = 1.0 - (nnz / total_elements)
            
            sparse_layers[name] = {
                'matrix': sparse_matrix,
                'bias': bias,
                'type': 'linear',
                'shape': masked_weight.shape,
                'on_gpu': use_gpu,
                "non_zero_elements": nnz,
                "total_elements": total_elements,
                "actual_sparsity": actual_sparsity
            }
            print(f"Layer {name}: shape={masked_weight.shape}, nnz={nnz}/{total_elements}, "
                  f"sparsity={actual_sparsity*100:.2f}%, on_gpu={use_gpu}")
            print('=' * 60)
        elif isinstance(module, MaskedConv2d):
            # For Conv2d, we'll flatten to 2D for CSR (simplified approach)
            weight = module.conv.weight.data.cpu().numpy()
            mask = module.mask.cpu().numpy()
            masked_weight = weight * mask
            
            # Flatten conv weights: [out_channels, in_channels, H, W] -> [out_channels, in_channels*H*W]
            out_ch, in_ch, h, w = masked_weight.shape
            flattened = masked_weight.reshape(out_ch, in_ch * h * w)
            
            # Convert to CSR format using CuPy (GPU or CPU)
            flattened_cp = cp.asarray(flattened)
            sparse_matrix = cp_sparse.csr_matrix(flattened_cp)
            
            # Handle bias
            if module.conv.bias is not None:
                bias = cp.asarray(module.conv.bias.data.cpu().numpy())
            else:
                bias = None
            
            sparse_layers[name] = {
                'matrix': sparse_matrix,
                'bias': bias,
                'type': 'conv2d',
                'shape': masked_weight.shape,
                'kernel_size': (h, w),
                'padding': module.conv.padding,
                'on_gpu': use_gpu
            }
    
    return sparse_layers


def forward_sparse_csr_linear(x, sparse_layers, layer_names):
    """
    Perform forward pass through linear layers using CSR sparse matrices.
    
    Args:
        x: Input tensor (numpy array), shape [batch_size, features]
        sparse_layers: Dictionary of sparse layers
        layer_names: List of layer names to process in order
        
    Returns:
        Output tensor (numpy array)
    """
    # Determine if we're using GPU based on first layer
    use_gpu = False
    if layer_names and layer_names[0] in sparse_layers:
        use_gpu = sparse_layers[layer_names[0]].get('on_gpu', False)
    
    # Convert input to CuPy array (GPU or CPU)
    if use_gpu:
        current = cp.asarray(x)
    else:
        current = cp.asarray(np.array(x))
    
    for layer_name in layer_names:
        if layer_name not in sparse_layers:
            continue
            
        layer_info = sparse_layers[layer_name]
        sparse_matrix = layer_info['matrix']
        
        if layer_info['type'] == 'linear':
            if current.ndim == 1:
                current = current.reshape(1, -1)
            
            result = sparse_matrix.dot(current.T).T  # [batch, out_features]
            
            if layer_info['bias'] is not None:
                result = result + layer_info['bias']
            
            current = result
            
            # Apply ReLU (except for last layer)
            if layer_name != layer_names[-1]:
                current = cp.maximum(0, current)
    
    # Convert back to numpy array
    return cp.asnumpy(current)


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Zero out gradients for masked weights (important!)
        for module in model.modules():
            if isinstance(module, (MaskedConv2d, MaskedLinear)):
                if isinstance(module, MaskedConv2d):
                    module.conv.weight.grad *= module.mask
                else:
                    module.linear.weight.grad *= module.mask
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, device, criterion=None):
    """Evaluate model - returns accuracy and optionally loss"""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Calculate loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader) if criterion is not None else None
    
    if criterion is not None:
        return accuracy, avg_loss
    else:
        return accuracy


# ============================================================================
# Benchmarking Functions
# ============================================================================

def benchmark_inference(model, dataloader, device, num_samples=1):
    """
    Benchmark inference time for different implementations.
    
    Args:
        model: Model to benchmark
        dataloader: DataLoader for test data
        device: Device to run on
        num_samples: Number of samples to benchmark
        
    Returns:
        Dictionary with timing results
    """
    results = {}
    
    # Get a batch of data
    inputs, _ = next(iter(dataloader))
    inputs = inputs[:num_samples].to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark Dense+Mask on GPU (if available)
    if device.type == 'cuda':
        model.eval()
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = model(inputs)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        results['Dense+Mask (GPU)'] = {
            'mean': np.mean(times) * 1000,  # Convert to ms
            'std': np.std(times) * 1000
        }
    
    # Benchmark Dense+Mask on CPU
    model_cpu = copy.deepcopy(model).cpu()
    inputs_cpu = inputs.cpu()
    model_cpu.eval()
    times = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model_cpu(inputs_cpu)
        times.append(time.time() - start)
    results['Dense+Mask (CPU)'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000
    }
    
    # Benchmark Truly Sparse (CSR) on CPU
    print("\n" + "="*60)
    print("Converting to CSR sparse format...")
    print("="*60)
    sparse_layers_cpu = convert_to_csr_sparse(model_cpu, use_gpu=False)
    
    # Check if matrices are actually sparse enough
    print("\nSparsity Check:")
    all_sparse_enough = True
    for name in sparse_layers_cpu:
        if 'actual_sparsity' in sparse_layers_cpu[name]:
            sparsity = sparse_layers_cpu[name]['actual_sparsity']
            if sparsity < 0.9:  # Less than 90% sparse
                print(f"⚠ WARNING: {name} is only {sparsity*100:.1f}% sparse - may not benefit from sparse ops")
                all_sparse_enough = False
            else:
                print(f"✓ {name}: {sparsity*100:.1f}% sparse")
    
    if num_samples == 1:
        print(f"\n⚠ WARNING: Benchmarking with only {num_samples} sample.")
        print("  Sparse operations benefit more from larger batches.")
        print("  Consider increasing num_samples for fair comparison.")
    
    linear_layer_order = ['masked_fc1', 'masked_fc2', 'masked_fc3', 'masked_fc4']
    
    # Simulate the conv layers output by using the masked model's conv output
    model_cpu.eval()
    with torch.no_grad():
        x = inputs_cpu
        x = model_cpu.pool(F.relu(model_cpu.masked_conv1(x)))
        x = model_cpu.pool(F.relu(model_cpu.masked_conv2(x)))
        conv_output_flat = x.view(x.size(0), -1)
    
    # Convert to numpy for sparse operations
    x_np = conv_output_flat.numpy()
    
    times = []
    for _ in range(10):
        start = time.time()
        output = forward_sparse_csr_linear(x_np, sparse_layers_cpu, linear_layer_order)
        times.append(time.time() - start)
    
    results['Truly Sparse CSR (CPU)'] = {
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000
    }
    
    # Benchmark Truly Sparse (CSR) on GPU (if GPU and CuPy GPU are available)
    if device.type == 'cuda' and CUPY_GPU_AVAILABLE:
        sparse_layers_gpu = convert_to_csr_sparse(model, use_gpu=True)
        
        # Get conv output on GPU
        model.eval()
        with torch.no_grad():
            x_gpu = inputs
            x_gpu = model.pool(F.relu(model.masked_conv1(x_gpu)))
            x_gpu = model.pool(F.relu(model.masked_conv2(x_gpu)))
            conv_output_flat_gpu = x_gpu.view(x_gpu.size(0), -1)
        
        x_gpu_np = conv_output_flat_gpu.cpu().numpy()
        
        # Warmup
        _ = forward_sparse_csr_linear(x_gpu_np, sparse_layers_gpu, linear_layer_order)
        if CUPY_GPU_AVAILABLE:
            cp.cuda.Stream.null.synchronize()
        
        times = []
        for _ in range(10):
            if CUPY_GPU_AVAILABLE:
                cp.cuda.Stream.null.synchronize()
            start = time.time()
            output = forward_sparse_csr_linear(x_gpu_np, sparse_layers_gpu, linear_layer_order)
            if CUPY_GPU_AVAILABLE:
                cp.cuda.Stream.null.synchronize()
            times.append(time.time() - start)
        
        results['Truly Sparse CSR (GPU)'] = {
            'mean': np.mean(times) * 1000,
            'std': np.std(times) * 1000
        }
    
    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_mask(mask, title="Weight Mask"):
    """Visualize a weight mask"""
    if len(mask.shape) == 4:  # Conv layer: [out_channels, in_channels, H, W]
        mask_2d = mask[0, 0].cpu().numpy()
    elif len(mask.shape) == 2:  # Linear layer: [out_features, in_features]
        mask_2d = mask.cpu().numpy()
    else:
        print(f"Cannot visualize mask of shape {mask.shape}")
        return
    
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_2d, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()


def plot_mask_evolution_per_epoch(sparsity_per_epoch, masks_per_epoch=None, save_path='plots/mask_evolution.png'):
    """Plot sparsity evolution per epoch for each layer (line plot)"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not sparsity_per_epoch:
        return
    
    # Get layer names from first epoch
    layer_names = list(sparsity_per_epoch[0].keys())
    epochs = range(len(sparsity_per_epoch))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for layer_name in layer_names:
        sparsities = [sparsity_per_epoch[e][layer_name] * 100 for e in epochs]
        ax.plot(epochs, sparsities, marker='o', label=layer_name, linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Sparsity (%)', fontsize=12)
    ax.set_title('Mask Evolution: Sparsity per Layer per Epoch', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved mask evolution plot to {save_path}")
    plt.close()


def plot_mask_2d_evolution(masks_per_epoch, save_path='plots/mask_2d_evolution.png'):
    """Plot 2D mask evolution as images for each layer per epoch"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not masks_per_epoch:
        return
    
    # Get layer names from first epoch
    layer_names = list(masks_per_epoch[0].keys())
    num_epochs = len(masks_per_epoch)
    
    # Create a separate figure for each layer
    for layer_name in layer_names:
        epochs_per_row = min(10, num_epochs)
        num_rows = (num_epochs + epochs_per_row - 1) // epochs_per_row
        
        fig, axes = plt.subplots(num_rows, epochs_per_row, figsize=(epochs_per_row * 2, num_rows * 2))
        if num_epochs == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(num_rows, epochs_per_row)
        
        def downsample_mask(mask, max_size=200):
            """Downsample mask to max_size x max_size for visualization"""
            mask = mask.copy()
            mask = (mask > 0.5).astype(float)
            
            if len(mask.shape) != 2:
                size = int(np.sqrt(mask.size))
                if size * size != mask.size:
                    pad_size = int(np.ceil(np.sqrt(mask.size)))**2
                    mask_flat = np.zeros(pad_size)
                    mask_flat[:mask.size] = mask.flatten()
                    mask = mask_flat.reshape(int(np.sqrt(pad_size)), int(np.sqrt(pad_size)))
                else:
                    mask = mask.reshape(size, size)
            
            h, w = mask.shape
            if h > max_size or w > max_size:
                step_h = max(1, h // max_size)
                step_w = max(1, w // max_size)
                mask = mask[::step_h, ::step_w]
                mask = (mask > 0.5).astype(float)
            
            return mask
        
        for epoch_idx in range(num_epochs):
            row = epoch_idx // epochs_per_row
            col = epoch_idx % epochs_per_row
            
            mask_2d = downsample_mask(masks_per_epoch[epoch_idx][layer_name])
            
            if epoch_idx == 0:
                rgb_image = np.zeros((*mask_2d.shape, 3))
                rgb_image[:, :, 0] = mask_2d
                rgb_image[:, :, 1] = mask_2d
                rgb_image[:, :, 2] = mask_2d
            else:
                prev_mask_2d = downsample_mask(masks_per_epoch[epoch_idx - 1][layer_name])
                
                if prev_mask_2d.shape != mask_2d.shape:
                    h_curr, w_curr = mask_2d.shape
                    h_prev, w_prev = prev_mask_2d.shape
                    h_new = min(h_prev, h_curr)
                    w_new = min(w_prev, w_curr)
                    prev_mask_2d = prev_mask_2d[:h_new, :w_new]
                    
                    if prev_mask_2d.shape != mask_2d.shape:
                        padded = np.zeros(mask_2d.shape)
                        padded[:prev_mask_2d.shape[0], :prev_mask_2d.shape[1]] = prev_mask_2d
                        prev_mask_2d = padded
            
                rgb_image = np.zeros((*mask_2d.shape, 3))
                
                newly_pruned = (prev_mask_2d == 1) & (mask_2d == 0)
                newly_regrown = (prev_mask_2d == 0) & (mask_2d == 1)
                unchanged_active = (prev_mask_2d == 1) & (mask_2d == 1)
                unchanged_pruned = (prev_mask_2d == 0) & (mask_2d == 0)
                
                rgb_image[newly_pruned] = [1.0, 0.0, 0.0]
                rgb_image[newly_regrown] = [0.0, 1.0, 0.0]
                rgb_image[unchanged_active] = [1.0, 1.0, 1.0]
                rgb_image[unchanged_pruned] = [0.0, 0.0, 0.0]
            
            ax = axes[row, col]
            im = ax.imshow(rgb_image, interpolation='nearest', aspect='auto')
            ax.set_title(f'Epoch {epoch_idx+1}\nSparsity: {(1-mask_2d.mean())*100:.1f}%', fontsize=7)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_epochs, num_rows * epochs_per_row):
            row = idx // epochs_per_row
            col = idx % epochs_per_row
            axes[row, col].axis('off')
        
        layer_type = 'Conv' if 'conv' in layer_name.lower() else 'FC'
        plt.suptitle(f'Mask Evolution: {layer_name} ({layer_type} Layer)\n'
                     f'Red=Newly Pruned | Green=Newly Regrown | White=Unchanged Active | Black=Unchanged Pruned', 
                     fontsize=12, fontweight='bold', y=0.98)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Newly Pruned'),
            Patch(facecolor='green', label='Newly Regrown'),
            Patch(facecolor='white', edgecolor='black', label='Unchanged Active'),
            Patch(facecolor='black', label='Unchanged Pruned')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8, 
                  bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        layer_save_path = save_path.replace('.png', f'_{layer_name}.png')
        plt.savefig(layer_save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved 2D mask evolution for {layer_name} to {layer_save_path}")
        plt.close()
    
    print(f"\n✓ Generated 2D mask evolution plots for {len(layer_names)} layers")


def plot_dense_vs_masked_comparison(dense_train_loss, dense_train_acc, dense_test_loss, dense_test_acc,
                                    masked_train_loss, masked_train_acc, masked_test_loss, masked_test_acc,
                                    save_path='plots/dense_vs_masked.png'):
    """Plot comparison of dense vs dense+mask training with test loss and differences"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(len(masked_train_loss))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Training Loss comparison
    axes[0].plot(epochs, dense_train_loss, 'b-o', label='Dense', linewidth=2, markersize=6)
    axes[0].plot(epochs, masked_train_loss, 'r-s', label='Dense+Mask', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Training Loss', fontsize=11)
    axes[0].set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Training Accuracy comparison
    axes[1].plot(epochs, dense_train_acc, 'b-o', label='Dense', linewidth=2, markersize=6)
    axes[1].plot(epochs, masked_train_acc, 'r-s', label='Dense+Mask', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Training Accuracy (%)', fontsize=11)
    axes[1].set_title('Training Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Test Loss comparison
    axes[2].plot(epochs, dense_test_loss, 'b-o', label='Dense', linewidth=2, markersize=6)
    axes[2].plot(epochs, masked_test_loss, 'r-s', label='Dense+Mask', linewidth=2, markersize=6)
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Test Loss', fontsize=11)
    axes[2].set_title('Test Loss Comparison', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Test Accuracy comparison
    axes[3].plot(epochs, dense_test_acc, 'b-o', label='Dense', linewidth=2, markersize=6)
    axes[3].plot(epochs, masked_test_acc, 'r-s', label='Dense+Mask', linewidth=2, markersize=6)
    axes[3].set_xlabel('Epoch', fontsize=11)
    axes[3].set_ylabel('Test Accuracy (%)', fontsize=11)
    axes[3].set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    # Loss Difference (Masked - Dense)
    loss_diff = [m - d for m, d in zip(masked_test_loss, dense_test_loss)]
    axes[4].plot(epochs, loss_diff, 'g-o', linewidth=2, markersize=6)
    axes[4].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[4].set_xlabel('Epoch', fontsize=11)
    axes[4].set_ylabel('Test Loss Difference (Masked - Dense)', fontsize=11)
    axes[4].set_title('Test Loss Difference', fontsize=12, fontweight='bold')
    axes[4].grid(True, alpha=0.3)
    axes[4].fill_between(epochs, 0, loss_diff, alpha=0.3, color='green' if loss_diff[-1] > 0 else 'red')
    
    # Accuracy Difference (Masked - Dense)
    acc_diff = [m - d for m, d in zip(masked_test_acc, dense_test_acc)]
    axes[5].plot(epochs, acc_diff, 'm-o', linewidth=2, markersize=6)
    axes[5].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[5].set_xlabel('Epoch', fontsize=11)
    axes[5].set_ylabel('Test Accuracy Difference (Masked - Dense) %', fontsize=11)
    axes[5].set_title('Test Accuracy Difference', fontsize=12, fontweight='bold')
    axes[5].grid(True, alpha=0.3)
    axes[5].fill_between(epochs, 0, acc_diff, alpha=0.3, color='green' if acc_diff[-1] > 0 else 'red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved dense vs masked comparison plot to {save_path}")
    plt.close()

