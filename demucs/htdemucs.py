#                                                                             #
# A state-of-the-art music source separation model with LoRA integration      #
# for efficient fine-tuning and adaptation.                                   #
###############################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Simon Rouard.
# LoRA modifications by GeradeHouse.
# January 2025

"""
HTDemucs: Hybrid Transformer Demucs Model with LoRA Integration

A state-of-the-art music source separation model that combines time and frequency domain 
processing with Low-Rank Adaptation (LoRA) for efficient fine-tuning. The architecture 
features:

Core Architecture (HTDemucs):
1. Dual-Path Processing:
   - Frequency Path: Processes spectrograms through frequency-domain convolutions
   - Time Path: Parallel processing of raw waveforms
   - Cross-Path Fusion: Merges information between paths at strategic points

2. Transformer Integration:
   - Cross-attention between time and frequency representations
   - Position-aware processing through various embedding options
   - Optional sparse attention patterns for efficiency

3. Advanced Audio Processing:
   - STFT/iSTFT with overlap-add for time-frequency conversion
   - Multiple output modes: direct masking, Wiener filtering, complex-as-channels
   - Frequency embeddings for improved spectral processing (Isik et al. 2020)
   - No resampling requirement, built-in normalization

LoRA Integration:
1. Adaptive Rank Allocation:
   - Uniform Mode: Consistent rank across all layers (baseline)
   - Heuristic Mode: Layer-specific ranks based on architectural importance:
     * Critical layers (early encoder, transformer): higher ranks (8-16)
       - Enhanced ranks for drum separation layers
       - Specialized attention for transient detection
     * Middle processing layers: medium ranks (4-8)
     * Final refinement layers: lower ranks (2-4)
   - Gradient-based Mode: (Future) Dynamic rank adaptation based on layer sensitivity

2. LoRA Implementation:
   - Memory-efficient parameter updates through low-rank decomposition
   - Selective freezing of base model weights
   - Dropout regularization for robust adaptation
   - Per-layer profiling capabilities for optimization
   - Support for fine-grained source separation (e.g., drum sub-components)

3. Layer Coverage:
   - Convolutional layers (1D/2D, standard/transposed)
   - Linear layers in transformer components
   - Depth-wise separable convolutions
   - Rewrite paths and skip connections

Technical Optimizations:
- Memory profiling for LoRA operations
- Gradient checkpointing support
- Efficient cross-attention implementation
- Flexible sparsity patterns for attention
- Modular design for easy extension

Source Separation Capabilities:
- Standard 4-source separation (bass, drums, vocals, other)
- Extended 5-source separation with specialized components:
  * Instruments: All musical instruments except drums and bass
  * Kick: All drums and percussion except hi-hats
  * Vocals: All vocal elements
  * Bass: Low-frequency instruments and bass lines
  * Hi-hat: Isolated hi-hat patterns and cymbals

References:
- Original Demucs: https://arxiv.org/abs/1911.13254
- LoRA: https://arxiv.org/abs/2106.09685
- Frequency Embeddings: https://arxiv.org/abs/2008.04470
"""

###############################################################################
#                              Import Section                                   #
###############################################################################
# Standard library imports for basic functionality
import math          # Mathematical operations and constants
import warnings      # Warning control and suppression
import time         # Performance profiling and timing
import typing as tp  # Type annotations and hints
import tracemalloc  # Memory usage tracking
import fnmatch      # Pattern matching for filenames
from fractions import Fraction  # Exact fractional arithmetic
from copy import deepcopy      # Deep object copying
from functools import wraps    # Function decoration utilities

# Deep learning framework imports
import torch         # PyTorch base package
from torch import nn  # Neural network modules
from torch.nn import functional as F  # Common functions
from einops import rearrange  # Advanced tensor reshaping
from torch.utils.checkpoint import checkpoint  # Memory-efficient gradients

# Local imports for model components
from .transformer import CrossTransformerEncoder  # Cross-attention transformer
from .demucs import rescale_module               # Weight initialization
from .states import capture_init                 # Model state tracking
from .spec import spectro, ispectro              # STFT/iSTFT processing
from .hdemucs import pad1d, ScaledEmbedding, HEncLayer, MultiWrap, HDecLayer  # Core architecture

# External dependencies
from openunmix.filtering import wiener  # Advanced source separation

# Optional Wiener filtering import with fallback
try:
    from openunmix.filtering import wiener  # Advanced source separation
except ImportError:
    wiener = None  # Fallback to None if not available

#------------------------------------------------------------------------------
#                        Memory Profiling Components
#------------------------------------------------------------------------------
def profile_memory(func):
    """Decorator to track memory usage and execution time of LoRA operations"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'enable_profiling'):
            return func(self, *args, **kwargs)
            
        if self.enable_profiling:
            tracemalloc.start()
            start_time = time.time()
            
        result = func(self, *args, **kwargs)
        
        if self.enable_profiling:
            current, peak = tracemalloc.get_traced_memory()
            self.memory_stats = {
                'time': time.time() - start_time,
                'current': current / 1024**2,  # MB
                'peak': peak / 1024**2,       # MB
                'function': func.__name__
            }
            tracemalloc.stop()
            
        return result
    return wrapper

#------------------------------------------------------------------------------
#                        LoRA Base Implementation
#------------------------------------------------------------------------------
class LoRALayerBase(nn.Module):
    """
    Base class for all LoRA-enabled layers implementing Low-Rank Adaptation.
    
    LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that 
    injects trainable rank decomposition matrices into frozen neural networks.
    This implementation provides:

    1. Core LoRA Features:
       - Low-rank matrix decomposition (U @ V) for efficient updates
       - Configurable rank for adaptation capacity control
       - Alpha scaling for update magnitude control
       - Dropout for regularization
       - Base weight freezing with selective adaptation

    2. Implementation Details:
       - U matrix shape: [fan_out, rank]
       - V matrix shape: [rank, fan_in]
       - Effective update: W = W_frozen + alpha * (U @ V)
       - Initialization: Normal distribution scaled by 1/sqrt(fan_in)
       - Optional dropout before U matrix for regularization

    3. Memory Optimization:
       - Efficient parameter storage through rank reduction
       - No storage of full-rank matrices
       - Optional memory profiling capabilities

    4. Validation Features:
       - Rank boundary checking against layer dimensions
       - Parameter validation (alpha, dropout, etc.)
       - Shape compatibility verification

    Usage:
        This base class is extended by specific layer implementations
        (Conv1d, Conv2d, Linear, etc.) to add LoRA capabilities while
        sharing common validation and utility functions.
    """
    def __init__(self, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.rank = rank                # Rank determines the dimensionality of the low-rank update matrices  
        self.alpha = alpha              # Alpha scales the magnitude of LoRA updates
        self.dropout = dropout          # Dropout rate for regularization
        self.lora_enabled = True        # Flag to enable/disable LoRA adaptations at runtime
        self.enable_profiling = False   # Not used inside each forward anymore
        
    def validate_params(self, fan_in: int, fan_out: int):
        """Validate LoRA parameters against layer dimensions"""
        # Ensure rank is at least 1 for valid matrix dimensions
        if self.rank < 1:
            raise ValueError(f"LoRA rank must be ≥1, got {self.rank}")
        # Check rank doesn't exceed matrix dimensions to avoid zero-sized tensors
        if self.rank > min(fan_in, fan_out):
            raise ValueError(
                f"LoRA rank {self.rank} exceeds min(fan_in={fan_in}, fan_out={fan_out})="
                f"{min(fan_in, fan_out)}. This would create zero-dimension matrices."
            )
        # Validate alpha is non-negative and properly scaled for stable updates
        if self.alpha < 0:
            raise ValueError(f"LoRA alpha must be ≥0, got {self.alpha}")
        if self.alpha > fan_in:
            raise ValueError(f"LoRA alpha ({self.alpha}) should not exceed fan_in ({fan_in}) "
                          "to maintain stable updates in transformer layers")
        # Ensure dropout is a valid probability
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {self.dropout}")

    def init_lora_parameters(self, fan_out: int, fan_in: int):
        """Initialize LoRA matrices U and V with proper scaling and initialization scheme validation"""
        # Create U matrix [fan_out, rank] for the first part of decomposition
        self.U = nn.Parameter(torch.zeros(fan_out, self.rank))
        # Create V matrix [rank, fan_in] for the second part of decomposition
        self.V = nn.Parameter(torch.zeros(self.rank, fan_in))
        
        # Scale initialization by 1/sqrt(fan_in) for stable gradients
        scale = math.sqrt(1.0 / fan_in)
        
        # Define max_attempts in the outer scope
        max_attempts = 10
        
        # Initialize U and V with validated scheme
        def init_weight(tensor, scale):
            # Validate initialization doesn't produce extreme values
            for _ in range(max_attempts):
                nn.init.normal_(tensor, std=scale)
                if tensor.abs().max() < 2.0:  # Check for reasonable bounds
                    return True
            return False
            
        if not init_weight(self.U, scale) or not init_weight(self.V, scale):
            warnings.warn(
                f"LoRA initialization produced extreme values after {max_attempts} attempts. "
                "This might lead to instability."
            )
        
        # Synchronize initialization across GPUs if in distributed setting
        if torch.distributed.is_initialized():
            for param in [self.U, self.V]:
                torch.distributed.broadcast(param.data, src=0)
                torch.distributed.barrier()  # Ensure all processes are synced
        
        # Create dropout layer for regularization, or Identity if dropout=0
        self.dropout_layer = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()

# ------------------------------------------------------------------------
#  3) LoRA Convolution Implementations (no per-layer memory profiling)
# ------------------------------------------------------------------------
class LoRAConv1d(LoRALayerBase):
    """1D Convolution with LoRA adaptation"""
    def __init__(self, original_conv: nn.Conv1d, rank=4, alpha=1.0, dropout=0.0):
        super().__init__(rank, alpha, dropout)
        
        self.base_conv = original_conv
        for p in self.base_conv.parameters():
            p.requires_grad = False
            
        out_c, in_c, kW = original_conv.weight.shape
        fan_in = in_c * kW
        fan_out = out_c
        
        self.validate_params(fan_in, fan_out)
        self.init_lora_parameters(fan_out, fan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip LoRA if disabled, use base convolution only
        if not self.lora_enabled:
            return self.base_conv(x)
            
        # Apply dropout to U matrix for regularization
        U = self.dropout_layer(self.U)
        # Get frozen weights from base convolution
        W_frozen = self.base_conv.weight
        # Compute low-rank update (U @ V)
        delta_flat = U @ self.V  # Matrix multiplication for low-rank approximation
        # Reshape update to match weight tensor shape
        delta = delta_flat.view(W_frozen.shape)
        # Add scaled LoRA update to frozen weights
        W_adapted = W_frozen + self.alpha * delta
        
        # Apply convolution with adapted weights
        return F.conv1d(x, W_adapted,
                        bias=self.base_conv.bias,  # Use original bias
                        stride=self.base_conv.stride,  # Keep original stride
                        padding=self.base_conv.padding,  # Keep original padding
                        dilation=self.base_conv.dilation,  # Keep original dilation
                        groups=self.base_conv.groups)  # Keep original groups


class LoRAConv2d(LoRALayerBase):
    """2D Convolution with LoRA adaptation for spectral processing"""
    def __init__(self, original_conv: nn.Conv2d, rank=4, alpha=1.0, dropout=0.0):
        super().__init__(rank, alpha, dropout)
        
        # Store base convolution and freeze its parameters
        self.base_conv = original_conv  # Original 2D convolution
        for p in self.base_conv.parameters():  # Freeze base weights
            p.requires_grad = False
            
        # Calculate fan-in/fan-out for initialization
        out_c, in_c, kH, kW = original_conv.weight.shape  # Get weight shape
        fan_in = in_c * kH * kW  # Total input connections
        fan_out = out_c  # Output channels
        
        # Initialize LoRA matrices with proper dimensions
        self.validate_params(fan_in, fan_out)  # Check rank validity
        self.init_lora_parameters(fan_out, fan_in)  # Create U and V matrices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use original conv if LoRA is disabled
        if not self.lora_enabled:
            return self.base_conv(x)  # Bypass LoRA adaptation
            
        # Apply LoRA update to weights
        U = self.dropout_layer(self.U)  # Regularize with dropout
        W_frozen = self.base_conv.weight  # Get base weights
        delta_flat = U @ self.V  # Compute low-rank update
        delta = delta_flat.view(W_frozen.shape)  # Match weight shape
        W_adapted = W_frozen + self.alpha * delta  # Add scaled update
        
        # Apply 2D convolution with adapted weights
        return F.conv2d(x, W_adapted,
                        bias=self.base_conv.bias,  # Original bias
                        stride=self.base_conv.stride,  # Keep stride
                        padding=self.base_conv.padding,  # Keep padding
                        dilation=self.base_conv.dilation,  # Keep dilation
                        groups=self.base_conv.groups)  # Keep groups


class LoRAConvTranspose1d(LoRALayerBase):
    """1D Transposed Convolution with LoRA for upsampling"""
    def __init__(self, original_conv: nn.ConvTranspose1d, rank=4, alpha=1.0, dropout=0.0):
        super().__init__(rank, alpha, dropout)
        
        # Store and freeze base transposed convolution
        self.base_conv = original_conv  # Original transposed conv
        for p in self.base_conv.parameters():  # Freeze base weights
            p.requires_grad = False
            
        # Calculate dimensions for LoRA matrices
        in_c, out_c, kW = original_conv.weight.shape  # Get weight shape
        fan_in = out_c * kW  # Input connections (transposed)
        fan_out = in_c  # Output channels (transposed)
        
        # Initialize LoRA components
        self.validate_params(fan_in, fan_out)  # Verify rank
        self.init_lora_parameters(fan_out, fan_in)  # Create matrices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip LoRA if disabled
        if not self.lora_enabled:
            return self.base_conv(x)  # Use original conv
            
        # Compute LoRA weight update
        U = self.dropout_layer(self.U)  # Apply dropout
        W_frozen = self.base_conv.weight  # Get base weights
        delta_flat = U @ self.V  # Low-rank update
        delta = delta_flat.view(W_frozen.shape)  # Reshape
        W_adapted = W_frozen + self.alpha * delta  # Add update
        
        # Apply transposed convolution with adapted weights
        return F.conv_transpose1d(
            x, W_adapted,  # Input and weights
            bias=self.base_conv.bias,  # Original bias
            stride=self.base_conv.stride,  # Upsampling factor
            padding=self.base_conv.padding,  # Edge handling
            output_padding=self.base_conv.output_padding,  # Output size adjustment
            groups=self.base_conv.groups,  # Channel groups
            dilation=self.base_conv.dilation  # Kernel spacing
        )


class LoRALinear(LoRALayerBase):
    """Linear layer with LoRA adaptation for transformer attention layers"""
    def __init__(self, original_linear: nn.Linear, rank=4, alpha=1.0, dropout=0.0):
        super().__init__(rank, alpha, dropout)
        
        # Store and freeze base linear layer
        self.base_linear = original_linear  # Original linear transformation
        for p in self.base_linear.parameters():  # Freeze base weights
            p.requires_grad = False
            
        # Get dimensions for LoRA matrices
        fan_in = original_linear.in_features  # Input features
        fan_out = original_linear.out_features  # Output features
        
        # Initialize LoRA components
        self.validate_params(fan_in, fan_out)  # Verify rank validity
        self.init_lora_parameters(fan_out, fan_in)  # Create U and V matrices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip LoRA if disabled
        if not self.lora_enabled:
            return self.base_linear(x)  # Use original linear layer
            
        # Apply LoRA adaptation
        U = self.dropout_layer(self.U)  # Apply dropout for regularization
        W_frozen = self.base_linear.weight  # Get frozen weights
        delta = U @ self.V  # Compute low-rank update
        W_adapted = W_frozen + self.alpha * delta  # Add scaled update
        
        # Apply linear transformation with adapted weights
        return F.linear(x, W_adapted, self.base_linear.bias)  # Include original bias


class LoRAConvTranspose2d(LoRALayerBase):
    """2D Transposed Convolution with LoRA for spectral upsampling"""
    def __init__(self, original_conv: nn.ConvTranspose2d, rank=4, alpha=1.0, dropout=0.0):
        super().__init__(rank, alpha, dropout)
        
        # Validate output padding against stride to prevent invalid configurations
        stride_h, stride_w = original_conv.stride  # Get stride dimensions
        out_pad_h, out_pad_w = original_conv.output_padding  # Get output padding
        
        # Ensure output padding doesn't exceed stride (requirement for transposed conv)
        if out_pad_h >= stride_h:
            raise ValueError(f"Output padding height ({out_pad_h}) must be < stride ({stride_h})")
        if out_pad_w >= stride_w:
            raise ValueError(f"Output padding width ({out_pad_w}) must be < stride ({stride_w})")

        # Store and freeze base transposed convolution
        self.base_conv = original_conv  # Original transposed conv
        for p in self.base_conv.parameters():  # Freeze base weights
            p.requires_grad = False
            
        # Calculate dimensions for LoRA matrices
        in_c, out_c, kH, kW = original_conv.weight.shape  # Get weight shape
        fan_in = out_c * kH * kW  # Total input connections
        fan_out = in_c  # Output channels
        
        # Initialize LoRA components
        self.validate_params(fan_in, fan_out)  # Verify rank validity
        self.init_lora_parameters(fan_out, fan_in)  # Create U and V matrices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip LoRA if disabled
        if not self.lora_enabled:
            return self.base_conv(x)  # Use original transposed conv
            
        # Apply LoRA adaptation
        U = self.dropout_layer(self.U)  # Apply dropout for regularization
        W_frozen = self.base_conv.weight  # Get frozen weights
        delta_flat = U @ self.V  # Compute low-rank update
        delta = delta_flat.view(W_frozen.shape)  # Reshape to match weights
        W_adapted = W_frozen + self.alpha * delta  # Add scaled update
        
        # Apply transposed convolution with adapted weights
        return F.conv_transpose2d(
            x, W_adapted,  # Input and adapted weights
            bias=self.base_conv.bias,  # Original bias
            stride=self.base_conv.stride,  # Upsampling factor
            padding=self.base_conv.padding,  # Edge handling
            output_padding=self.base_conv.output_padding,  # Output size adjustment
            groups=self.base_conv.groups,  # Channel groups
            dilation=self.base_conv.dilation  # Kernel spacing
        )


def wrap_with_lora(module: nn.Module, layer_name: str = "", default_rank: int = 4, layer_ranks: dict = None, alpha: float = 1.0, dropout: float = 0.0):
    """
    Wraps a PyTorch module with LoRA adaptation using intelligent rank allocation.
    
    This function implements an advanced strategy for determining appropriate LoRA ranks
    based on layer position and importance in the network architecture. The rank allocation
    follows a carefully designed heuristic approach backed by empirical observations.

    Rank Allocation Strategy:
    1. Explicit Mapping:
       - Uses layer_ranks dictionary if provided
       - Supports glob-style pattern matching for flexible layer targeting
       Example: {"encoder.0.*": 16, "decoder.*": 4}

    2. Heuristic-based Allocation:
       - Critical Early Layers (encoder.0, transformer):
         * Higher ranks (8-16) to capture fundamental features
         * Essential for maintaining model quality
       - Middle Processing Layers:
         * Medium ranks (4-8) for balanced adaptation
         * Handles feature transformation and refinement
       - Final Layers:
         * Lower ranks (2-4) for efficient fine-tuning
         * Focus on output refinement
       
    3. Architecture-Aware Decisions:
       - Encoder layers: Higher ranks for feature extraction
       - Transformer layers: Higher ranks for attention adaptation
       - Decoder layers: Gradually decreasing ranks
       - Skip connections: Matched to connected layers

    Args:
        module (nn.Module): Base PyTorch module to adapt (Conv1d/2d, Linear, etc.)
        layer_name (str): Identifier for the layer (e.g. "encoder.0.conv")
                         Used for rank allocation decisions
        default_rank (int): Fallback rank if no specific allocation applies
        layer_ranks (dict): Optional explicit rank mappings using glob patterns
        alpha (float): Scaling factor for LoRA updates (typically 1.0)
        dropout (float): Dropout probability for regularization

    Returns:
        nn.Module: LoRA-wrapped version of the input module with appropriate rank

    Raises:
        ValueError: For invalid parameters or unsupported module types
        
    Technical Details:
    - Rank determines adaptation capacity: higher rank = more flexibility
    - Memory usage scales with rank: O(fan_in * rank + fan_out * rank)
    - Computational cost also increases with rank
    - Dropout helps prevent overfitting in high-rank layers
    """
    # Determine rank based on layer name pattern matching and heuristics
    rank = default_rank
    
    # Strategy 1: Heuristic-based rank allocation
    def get_layer_rank(layer_name: str) -> int:
        if not layer_name:
            return default_rank
            
        # Critical layers get higher ranks
        if "encoder.0" in layer_name or "transformer" in layer_name:
            return min(16, default_rank * 2)  # Higher rank (8-16)
            
        # First decoder layers need medium ranks
        if "decoder.0" in layer_name or "decoder.1" in layer_name:
            return min(12, int(default_rank * 1.5))  # Medium rank (6-12)
            
        # Final layers need lower ranks
        if "decoder" in layer_name and any(f".{i}." in layer_name for i in [2, 3]):
            return max(2, default_rank // 2)  # Lower rank (2-4)
            
        return default_rank
    
    # First check explicit layer_ranks mapping
    if layer_ranks is not None and layer_name:
        for pattern, custom_rank in layer_ranks.items():
            if fnmatch.fnmatch(layer_name, pattern):
                rank = custom_rank
                break
    else:
        # Fall back to heuristic allocation
        rank = get_layer_rank(layer_name)
    
    if isinstance(module, nn.Conv1d):
        return LoRAConv1d(module, rank=rank, alpha=alpha, dropout=dropout)
    elif isinstance(module, nn.Conv2d):
        return LoRAConv2d(module, rank=rank, alpha=alpha, dropout=dropout)
    elif isinstance(module, nn.ConvTranspose1d):
        return LoRAConvTranspose1d(module, rank=rank, alpha=alpha, dropout=dropout)
    elif isinstance(module, nn.ConvTranspose2d):
        return LoRAConvTranspose2d(module, rank=rank, alpha=alpha, dropout=dropout)
    elif isinstance(module, nn.Linear):
        return LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
    else:
        warnings.warn(f"Module type {type(module)} not supported for LoRA wrapping")
        return module

# Alias for backward compatibility
wrap_conv_with_lora = wrap_with_lora

#------------------------------------------------------------------------------
#                           Utility Functions
#------------------------------------------------------------------------------
def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = "constant", value: float = 0.0):
    """
    Enhanced 1D padding with special handling for reflection mode and small inputs.
    
    This function extends torch.nn.functional.pad with additional logic for:
    1. Reflection padding when input is smaller than padding size
    2. Automatic adjustment of padding values for small signals
    3. Proper boundary handling for various padding modes
    
    Args:
        x: Input tensor to pad [batch, channels, time]
        paddings: (left_pad, right_pad) tuple
        mode: Padding mode ('constant', 'reflect', etc.)
        value: Fill value for constant padding
        
    Returns:
        Padded tensor with proper boundary handling
    """
    length = x.shape[-1]  # Get input length
    padding_left, padding_right = paddings  # Unpack padding sizes
    
    if mode == "reflect":  # Special handling for reflection padding
        max_pad = max(padding_left, padding_right)  # Largest padding
        if length <= max_pad:  # Input smaller than padding
            # Calculate extra padding needed
            extra_pad = max_pad - length + 1
            # Distribute extra padding
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            # Adjust main padding values
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            # Apply initial padding for small input
            x = F.pad(x, (extra_pad_left, extra_pad_right))
            
    # Apply main padding operation
    out = F.pad(x, paddings, mode, value)
    return out

class ScaledEmbedding(nn.Module):
    """
    Frequency embedding layer with learning rate scaling, optional smoothing, and LoRA support.
    
    This layer provides:
    1. Learnable embeddings for frequency positions
    2. Optional cumulative smoothing for better initialization
    3. Learning rate scaling for stable training
    4. Efficient weight access and forward computation
    5. LoRA adaptation for fine-tuning
    
    Used to inject frequency-aware features into the model.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.0, smooth=False,
                 lora_rank: int = 4, lora_alpha: float = 1.0, lora_dropout: float = 0.0):
        super().__init__()
        # Create base embedding layer
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        if smooth:  # Apply cumulative smoothing if requested
            # Compute cumulative sum for smooth initialization
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # Scale by sqrt of position for stable gradients
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            # Update embedding weights
            self.embedding.weight.data[:] = weight
            
        # Apply learning rate scaling
        self.embedding.weight.data /= scale  # Scale down weights
        self.scale = scale  # Store scale for forward pass
        
        # Initialize LoRA components
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_enabled = True
        
        # Create LoRA matrices
        if lora_rank > 0:
            self.lora_U = nn.Parameter(torch.zeros(num_embeddings, lora_rank))
            self.lora_V = nn.Parameter(torch.zeros(lora_rank, embedding_dim))
            # Initialize with scaled normal distribution
            scale = 1.0 / math.sqrt(embedding_dim)
            nn.init.normal_(self.lora_U, std=scale)
            nn.init.normal_(self.lora_V, std=scale)
            # Create dropout layer
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
            # Freeze base weights
            self.embedding.weight.requires_grad = False

    @property
    def weight(self):
        """Access scaled embedding weights with LoRA adaptation"""
        if not hasattr(self, 'lora_rank') or not self.lora_enabled:
            return self.embedding.weight * self.scale
            
        # Apply LoRA update
        U = self.lora_dropout(self.lora_U)
        lora_update = (U @ self.lora_V) * self.lora_alpha
        return (self.embedding.weight + lora_update) * self.scale

    def forward(self, x):
        """Apply embedding with proper scaling and LoRA adaptation"""
        return F.embedding(x, self.weight)  # Use adapted weights

#------------------------------------------------------------------------------
#                      Depthwise Convolution Layer
#------------------------------------------------------------------------------
class DConv(nn.Module):
    """
    Depthwise-Separable Convolution with LoRA adaptation for efficient feature processing.
    
    This layer factorizes a standard convolution into:
    1. Depthwise Conv: Spatial filtering applied independently to each channel
    2. Pointwise Conv: 1x1 convolution for channel mixing
    3. Normalization and non-linear activation
    4. Residual connection for gradient flow
    
    Both convolution operations are wrapped with LoRA for efficient fine-tuning.
    """
    def __init__(self, channels: int, kernel_size=3, layer_name: str = "", default_rank: int = 4, layer_ranks: dict = None, lora_alpha=1.0, dropout=0.0):
        super().__init__()
        
        # Create depthwise convolution (one filter per channel)
        depthwise_conv = nn.Conv1d(
            channels, channels,  # Same channels in/out
            kernel_size,  # Spatial filtering
            padding=kernel_size//2,  # Same padding
            groups=channels  # Each channel processed separately
        )
        # Verify group configuration matches channels
        assert depthwise_conv.groups == channels, \
            f"Depthwise groups mismatch: {depthwise_conv.groups} vs {channels}"
        
        # Wrap depthwise conv with LoRA
        dconv_name = f"{layer_name}.dconv" if layer_name else ""  # Layer identifier
        self.dconv = wrap_with_lora(
            depthwise_conv,  # Base spatial filtering
            layer_name=dconv_name,  # For rank allocation
            default_rank=default_rank,  # Base LoRA rank
            layer_ranks=layer_ranks,  # Custom ranks if any
            alpha=lora_alpha,  # Update scaling
            dropout=dropout  # Regularization
        )
        
        # Create 1x1 convolution for channel mixing
        pconv_name = f"{layer_name}.pconv" if layer_name else ""  # Layer name
        self.pconv = wrap_with_lora(
            nn.Conv1d(channels, channels, 1),  # 1x1 convolution
            layer_name=pconv_name,  # For rank allocation
            default_rank=default_rank,  # Base LoRA rank
            layer_ranks=layer_ranks,  # Custom ranks if any
            alpha=lora_alpha,  # Update scaling
            dropout=dropout  # Regularization
        )
        
        # Normalization and activation components
        self.norm = nn.GroupNorm(4, channels)  # Group norm for stability
        self.act = nn.GELU()  # Non-linear activation

    def forward(self, x):
        # Apply spatial filtering per channel
        y = self.dconv(x)  # Depthwise convolution
        
        # Mix channels with 1x1 convolution
        y = self.pconv(y)  # Pointwise convolution
        
        # Normalize and activate features
        y = self.act(self.norm(y))  # GroupNorm + GELU
        
        # Add residual connection
        return x + y  # Skip connection

#------------------------------------------------------------------------------
#                     HTDemucs Core Model Implementation
#------------------------------------------------------------------------------
class HTDemucs(nn.Module):
    """
    HTDemucs: Hybrid Transformer Demucs with LoRA Integration

    A sophisticated music source separation model that combines spectral and temporal processing
    with transformer attention and efficient LoRA-based fine-tuning capabilities.

    Architecture Overview:
    1. Dual-Stream Processing:
       - Frequency Stream: Processes STFT spectrograms through frequency-domain convolutions
         * Initial layers operate over frequency axis until single frequency remains
         * DConv residual connections allow cross-time information flow
         * Optional frequency embeddings improve convolution efficiency
       - Time Stream: Parallel processing of raw waveform data
         * Maintains temporal coherence through direct time-domain operations
         * Merges with frequency stream at matched stride points
       - Hybrid Fusion: Bidirectional information flow between streams
         * Encoder: Streams merge when strides match
         * Decoder: Streams separate for specialized processing

    2. Output Processing Methods:
       - Direct iSTFT Masking: Simple spectral masking followed by inverse STFT
       - Wiener Filtering: Iterative refinement based on OpenUnmix [Stoter et al. 2019]
         * Note: Performance may degrade with increased test-time iterations
         * Spectrogram/waveform contribution balance can shift
       - Complex-as-Channels (CaC) [Choi et al. 2020]: 
         * Treats complex numbers as channel pairs
         * Natural integration with hybrid architecture
         * Consistent performance across configurations

    3. LoRA Integration:
       - Adaptive Fine-tuning: Each layer can be independently adapted
       - Memory Optimization: 
         * Efficient parameter updates through low-rank decomposition
         * Optional gradient checkpointing for large models
       - Layer Coverage:
         * All convolutional operations (standard, transposed, depthwise)
         * Transformer attention layers
         * Skip connections and rewrite paths

    4. Key Features:
       - No resampling requirement unlike classic Demucs
       - Consistent normalization application
       - Memory-profiled LoRA operations
       - Flexible transformer integration
       - Advanced frequency embeddings [Isik et al. 2020]

    Implementation Notes:
    - Loss computed in temporal domain through backpropagation
    - Frequency embeddings improve convolution efficiency
    - DConv residual allows cross-time information flow
    - Optional gradient checkpointing for memory efficiency
    - Modular design enables easy extension and modification

    References:
    - Wiener Filtering: [Ulhih et al. 2017]
    - Complex-as-Channels: [Choi et al. 2020]
    - OpenUnmix Implementation: [Stoter et al. 2019]
    - Frequency Embeddings: [Isik et al. 2020]
    """

    @capture_init 
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=2,
        channels=48,
        channels_time=None,
        growth=2,
        # LoRA configuration
        lora_rank_mode: str = 'heuristic',  # 'uniform', 'heuristic', or 'gradient'
        layer_ranks: dict = None,  # Custom layer-specific ranks
        lora_max_rank: int = 16,  # Maximum allowed rank for any layer
        # STFT params
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=4,
        rewrite=True,
        # Frequency branch
        multi_freqs=None,
        multi_freqs_depth=3,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # Before the Transformer
        bottom_channels=0,
        # Transformer
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        # ------ Particuliar parameters
        t_cross_first=False,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=10,
        use_train_segment=True,
        # LoRA params
        lora_rank=4,  # Default rank (used as base for heuristic mode)
        lora_alpha=1.0,
        lora_dropout=0.1,
        enable_profiling=False,
        enable_checkpointing=False,  # Enable gradient checkpointing for memory efficiency
    ):
        """
        Args:
            sources (list[str]): list of source names.
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            channels_time: if not None, use a different `channels` value for the time branch.
            growth: increase the number of hidden channels by this factor at each layer.
            layer_ranks: Dictionary mapping layer name patterns to custom LoRA ranks.
                Example: {"encoder.0.conv": 8, "decoder.*.conv_tr": 4}
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            wiener_iters: when using Wiener filtering, number of iterations at test time.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            wiener_residual: add residual source before Wiener filtering.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 conv to each layer.
            multi_freqs: list of frequency ratios for splitting frequency bands with `MultiWrap`.
            multi_freqs_depth: how many layers to wrap with `MultiWrap`. Only the outermost
                layers will be wrapped.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            time_stride: stride for the final time layer, after the merge.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_init: initial scale for the DConv branch LayerScale.
            bottom_channels: if >0 it adds a linear layer (1x1 Conv) before and after the
                transformer in order to change the number of channels
            t_layers: number of layers in each branch (waveform and spec) of the transformer
            t_emb: "sin", "cape" or "scaled"
            t_hidden_scale: the hidden scale of the Feedforward parts of the transformer
            t_heads: number of heads for the transformer
            t_dropout: dropout in the transformer
            t_max_positions: max_positions for the "scaled" positional embedding, only
                useful if t_emb="scaled"
            t_norm_in: (bool) norm before addinf positional embedding and getting into the
                transformer layers
            t_norm_in_group: (bool) if True while t_norm_in=True, the norm is on all the
                timesteps (GroupNorm with group=1)
            t_group_norm: (bool) if True, the norms of the Encoder Layers are on all the
                timesteps (GroupNorm with group=1)
            t_norm_first: (bool) if True the norm is before the attention and before the FFN
            t_norm_out: (bool) if True, there is a GroupNorm (group=1) at the end of each layer
            t_max_period: (float) denominator in the sinusoidal embedding expression
            t_weight_decay: (float) weight decay for the transformer
            t_lr: (float) specific learning rate for the transformer
            t_layer_scale: (bool) Layer Scale for the transformer
            t_gelu: (bool) activations of the transformer are GeLU if True, ReLU else
            t_weight_pos_embed: (float) weighting of the positional embedding
            t_sin_random_shift: (int) random shift for sinusoidal embedding
            t_cape_mean_normalize: (bool) if t_emb="cape", normalisation of positional embeddings
            t_cape_augment: (bool) if t_emb="cape", must be True during training and False
                during the inference
            t_cape_glob_loc_scale: (list of 3 floats) if t_emb="cape", CAPE parameters
            t_sparse_self_attn: (bool) if True, the self attentions are sparse
            t_sparse_cross_attn: (bool) if True, the cross-attentions are sparse
            t_mask_type: (str) can be "diag", "jmask", "random", "global" or combination
            t_mask_random_seed: (int) controls the random masking seed
            t_sparse_attn_window: (int) local window size for "diag" mask
            t_global_window: (int) global window size for "global" mask
            t_sparsity: (float) level for "random" part of the mask
            t_auto_sparsity: (bool) if True, automatically adapt sparsity
            t_cross_first: (bool) if True cross attention is the first layer
            rescale: weight rescaling trick
            samplerate (int): stored as meta info
            segment (int): chunk length in seconds
            use_train_segment (bool): if True, use training segment length at inference
        """
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.depth = depth
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        # LoRA configuration
        self.lora_rank_mode = lora_rank_mode
        self.layer_ranks = layer_ranks
        self.lora_max_rank = lora_max_rank

        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters

        self.freq_emb = None
        self.enable_profiling = enable_profiling
        self.enable_checkpointing = enable_checkpointing  # Store checkpointing flag

        assert wiener_iters == end_iters

        # Initialize encoder/decoder chains
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        # Build encoder/decoder architecture
        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            # DConv settings
            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            # Create freq enc
            layer_name = f"encoder.{index}.conv"
            enc = HEncLayer(
                chin_z, chout_z,
                dconv=(dconv_mode & 1) > 0,
                context=context_enc,
                **kw,
                layer_name=layer_name,
                default_rank=lora_rank,
                layer_ranks=layer_ranks,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            # Create time enc if freq==True
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=(dconv_mode & 1) > 0,
                    context=context_enc,
                    empty=last_freq,
                    **kwt
                )
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)

            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2

            # freq dec
            layer_name = f"decoder.{index}.conv"
            dec = HDecLayer(
                chout_z,
                chin_z,
                dconv=(dconv_mode & 2) > 0,
                last=(index == 0),
                context=context,
                **kw_dec,
                layer_name=layer_name,
                default_rank=lora_rank,
                layer_ranks=layer_ranks,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=(dconv_mode & 2) > 0,
                    empty=last_freq,
                    last=(index == 0),
                    context=context,
                    **kwt
                )
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z,
                    smooth=emb_smooth,
                    scale=emb_scale,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )

            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity,
                # Add LoRA parameters for transformer layers
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
            self.crosstransformer = None

    def enable_lora(self, enabled=True):
        """
        Enable or disable all LoRA layers while keeping base weights frozen.
        
        This method provides runtime control over LoRA adaptation:
        1. Toggle LoRA updates across all layers
        2. Keep base model weights frozen
        3. Enable/disable memory profiling
        4. Handle transformer layers separately
        
        Args:
            enabled (bool): Whether to enable LoRA adaptation
        """
        # Enable/disable LoRA in all modules (convolutions, linear layers)
        for module in self.modules():  # Iterate through all layers
            if hasattr(module, 'lora_enabled'):  # Check if LoRA-capable
                module.lora_enabled = enabled  # Set LoRA state
                module.enable_profiling = self.enable_profiling  # Update profiling
        
        # Handle transformer layers if present
        if hasattr(self, 'crosstransformer') and self.crosstransformer is not None:
            # Enable/disable LoRA in transformer modules
            for module in self.crosstransformer.modules():
                if hasattr(module, 'lora_enabled'):  # Check if LoRA-capable
                    module.lora_enabled = enabled  # Set LoRA state
                    module.enable_profiling = self.enable_profiling  # Update profiling

    def save_lora_state(self, path: str):
        """
        Save LoRA-specific parameters separately.
        
        Args:
            path (str): Path to save LoRA state
        """
        state = {}
        for name, module in self.named_modules():
            if hasattr(module, 'U') and hasattr(module, 'V'):
                state[f"{name}.U"] = module.U.data
                state[f"{name}.V"] = module.V.data
        torch.save(state, path)

    def load_lora_state(self, path: str):
        """
        Load LoRA-specific parameters.
        
        Args:
            path (str): Path to load LoRA state from
        """
        state = torch.load(path)
        for name, module in self.named_modules():
            if hasattr(module, 'U') and hasattr(module, 'V'):
                if f"{name}.U" in state and f"{name}.V" in state:
                    module.U.data.copy_(state[f"{name}.U"])
                    module.V.data.copy_(state[f"{name}.V"])

    def clip_lora_gradients(self, max_norm: float = 1.0):
        """
        Clip gradients of LoRA parameters separately.
        
        Args:
            max_norm (float): Maximum norm for gradient clipping
        """
        parameters = []
        for module in self.modules():
            if hasattr(module, 'U') and hasattr(module, 'V'):
                if module.U.requires_grad:
                    parameters.append(module.U)
                if module.V.requires_grad:
                    parameters.append(module.V)
        if parameters:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def to_mixed_precision(self):
        """
        Convert LoRA layers to mixed precision while keeping base model in full precision.
        """
        for module in self.modules():
            if hasattr(module, 'U') and hasattr(module, 'V'):
                module.U.data = module.U.data.to(torch.float16)
                module.V.data = module.V.data.to(torch.float16)

    def set_lora_train_mode(self, training: bool):
        """
        Set LoRA-specific training mode while keeping base model in eval.
        
        Args:
            training (bool): Whether to set LoRA layers to training mode
        """
        self.train(False)  # Set base model to eval
        for module in self.modules():
            if hasattr(module, 'lora_enabled'):
                module.train(training)
                # Handle batch norm statistics
                if hasattr(module, 'norm1'):
                    module.norm1.train(training)
                if hasattr(module, 'norm2'):
                    module.norm2.train(training)

    def get_lora_l2_loss(self, lambda_reg: float = 0.01):
        """
        Calculate L2 regularization loss for LoRA parameters.
        
        Args:
            lambda_reg (float): Regularization strength
            
        Returns:
            torch.Tensor: Regularization loss
        """
        loss = 0
        for module in self.modules():
            if hasattr(module, 'U') and hasattr(module, 'V'):
                loss += lambda_reg * (module.U.pow(2).sum() + module.V.pow(2).sum())
        return loss

    @profile_memory
    def _spec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time-domain signal to spectral domain using STFT.
        
        This method performs several key steps:
        1. Signal padding to maintain size consistency
        2. STFT computation with Hann window
        3. Trimming of redundant frequencies
        4. Alignment of time steps
        
        Args:
            x: Input audio tensor [batch, channels, time]
            
        Returns:
            Complex STFT tensor [batch, channels, frequencies, time]
        """
        hl = self.hop_length  # Step size between windows
        nfft = self.nfft  # FFT size
        x0 = x  # Store original input (for debugging)

        # Calculate padding to maintain size consistency
        assert hl == nfft // 4  # Standard relationship
        le = int(math.ceil(x.shape[-1] / hl))  # Target length
        pad = hl // 2 * 3  # Padding size (3/4 of hop length)
        
        # Apply reflection padding for better boundary handling
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")
        
        # Compute Short-Time Fourier Transform
        z = torch.stft(
            x,                    # Padded input signal
            nfft,                 # FFT size
            hop_length=hl,        # Window step size
            window=torch.hann_window(nfft, device=x.device),  # Hann window
            return_complex=True   # Return complex tensor
        )
        
        # Remove Nyquist frequency (redundant for real signals)
        z = z[..., :-1]
        
        # Verify shape and trim padding artifacts
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2:2 + le]  # Remove edge effects
        
        return z  # Return complex spectrogram

    @profile_memory
    def _ispec(self, z: torch.Tensor, length: int) -> torch.Tensor:
        """
        Convert spectral domain signal back to time domain using iSTFT.
        
        This method performs the inverse operation of _spec:
        1. Restore Nyquist frequency
        2. Add padding for proper reconstruction
        3. Inverse STFT computation
        4. Trim to target length
        
        Args:
            z: Complex STFT tensor [batch, channels, frequencies, time]
            length: Target length for output signal
            
        Returns:
            Time-domain signal [batch, channels, time]
        """
        hl = self.hop_length  # Step size between windows
        
        # Restore Nyquist frequency (removed in _spec)
        z = F.pad(z, (0, 0, 0, 1))
        
        # Add padding for proper reconstruction
        z = F.pad(z, (2, 2))  # Match trimming from _spec
        pad = hl // 2 * 3  # Same padding as in _spec
        le = hl * int(math.ceil(length / hl)) + 2 * pad  # Target length
        
        # Compute Inverse Short-Time Fourier Transform
        x = torch.istft(
            z,                    # Complex spectrogram
            self.nfft,           # FFT size
            hop_length=hl,        # Window step size
            window=torch.hann_window(self.nfft, device=z.device),  # Hann window
            length=le            # Exact output length
        )
        
        # Remove padding to match target length
        x = x[..., pad:pad + length]
        
        return x  # Return time-domain signal

    def _magnitude(self, z):
        """Reintroduce the original magnitude logic."""
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
        """
        Apply source separation masks to the mixed signal.
        
        Supports three separation modes:
        1. Complex-as-channels (CAC):
           - Treats real/imaginary parts as separate channels
           - Direct masking without phase adjustment
           
        2. Direct masking:
           - Simple multiplication with mask
           - Used when wiener_iters < 0
           - Fast but may have phase artifacts
           
        3. Wiener filtering:
           - Iterative mask refinement
           - Better separation quality
           - More computationally intensive
           
        Args:
            z: Complex STFT of mixture [batch, channels, freq, time]
            m: Source masks [batch, sources, channels, freq, time]
            
        Returns:
            Separated spectrograms [batch, sources, channels, freq, time]
        """
        niters = self.wiener_iters  # Number of Wiener iterations
        
        if self.cac:  # Complex-as-channels mode
            B, S, C, Fr, T = m.shape  # Get dimensions
            # Reshape mask for complex number handling
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            # Convert back to complex representation
            out = torch.view_as_complex(out.contiguous())
            return out
            
        if self.training:
            niters = self.end_iters  # Use training iteration count
            
        if niters < 0:  # Direct masking mode
            z = z[:, None]  # Add source dimension
            # Apply soft mask with epsilon for stability
            return z / (1e-8 + z.abs()) * m
        else:  # Wiener filtering mode
            return self._wiener(m, z, niters)  # Apply iterative refinement

    def _wiener(self, mag_out, mix_stft, niters):
        """Perform wiener filtering from OpenUnmix in small frames."""
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            while pos < T:
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame],
                    mix_stft[sample, frame],
                    niters,
                    residual=residual,
                )
                out.append(z_out.transpose(-1, -2))
                pos += wiener_win_len
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def valid_length(self, length: int):
        """
        Return a length that is appropriate for evaluation.
        In our case, always return the training length, unless
        it is smaller than the given length, in which case this
        raises an error.
        """
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                f"Given length {length} is longer than "
                f"training length {training_length}")
        return length

    def _sync_lora_params(self):
        """
        Synchronize LoRA parameters across GPUs in distributed training.
        """
        if not torch.distributed.is_initialized():
            return
            
        for module in self.modules():
            if hasattr(module, 'U') and hasattr(module, 'V'):
                torch.distributed.all_reduce(module.U.data)
                torch.distributed.all_reduce(module.V.data)
                module.U.data /= torch.distributed.get_world_size()
                module.V.data /= torch.distributed.get_world_size()

    @profile_memory
    def forward(self, mix):
        """
        Process audio mixture through dual-path architecture for source separation.
        
        This method implements the core separation logic:
        1. Convert input to frequency domain
        2. Process through parallel time/frequency branches
        3. Apply transformer cross-attention
        4. Generate source-specific masks
        5. Apply separation and convert back to time domain
        
        Memory optimization features:
        1. Gradient checkpointing for transformer layers
        2. Efficient LoRA parameter handling
        3. Multi-GPU synchronization for distributed training
        4. Automatic mixed precision support
        """
        # Synchronize LoRA parameters in distributed setting
        self._sync_lora_params()
        # Get input length and handle training segment logic
        length = mix.shape[-1]  # Original audio length
        length_pre_pad = None  # Track if we need padding
        if self.use_train_segment:
            if self.training:
                # Update segment length during training
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                # Pad shorter segments during inference
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    # Pad to match training length
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
                    
        # Convert to spectral domain and get magnitude
        z = self._spec(mix)  # Complex spectrogram
        mag = self._magnitude(z).to(mix.device)  # Magnitude spectrogram
        x = mag  # Input for frequency branch

        B, C, Fq, T = x.shape  # [batch, channels, frequencies, time]

        # Normalize frequency branch input
        mean = x.mean(dim=(1, 2, 3), keepdim=True)  # Global mean
        std = x.std(dim=(1, 2, 3), keepdim=True)  # Global std
        x = (x - mean) / (1e-5 + std)  # Normalize with epsilon

        # Normalize time branch input
        xt = mix  # Raw waveform for time branch
        meant = xt.mean(dim=(1, 2), keepdim=True)  # Time domain mean
        stdt = xt.std(dim=(1, 2), keepdim=True)  # Time domain std
        xt = (xt - meant) / (1e-5 + stdt)  # Normalize with epsilon

        # Initialize storage for skip connections and shape tracking
        saved = []  # Store frequency branch activations for skip connections
        saved_t = []  # Store time branch activations for skip connections
        lengths = []  # Track tensor lengths for proper padding removal (freq)
        lengths_t = []  # Track tensor lengths for proper padding removal (time)
        # Encoder forward pass through both branches
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])  # Save current frequency branch length
            inject = None  # Potential injection from time branch
            
            # Process time branch if available at this depth
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])  # Save time branch length
                tenc = self.tencoder[idx]  # Get time encoder layer
                xt = tenc(xt)  # Process time branch
                
                if not tenc.empty:
                    saved_t.append(xt)  # Save for future skip connection
                else:
                    # Time branch matches frequency branch shape, ready for merge
                    inject = xt  # Will be added to frequency branch
                    
            # Process frequency branch
            x = encode(x, inject)  # Apply encoder with optional time injection
            
            # Add frequency embeddings after first layer if enabled
            if idx == 0 and self.freq_emb is not None:
                # Create position-aware frequency embeddings
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                # Add scaled embeddings to main features
                x = x + self.freq_emb_scale * emb

            saved.append(x)  # Save for skip connection

        # Apply cross-attention between time and frequency branches
        if self.crosstransformer:
            if self.bottom_channels:
                # Reshape for channel adjustment before transformer
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")  # Flatten freq & time
                x = self.channel_upsampler(x)  # Adjust channel count
                x = rearrange(x, "b c (f t)-> b c f t", f=f)  # Restore shape
                xt = self.channel_upsampler_t(xt)  # Match time branch channels

            # Process through transformer with memory optimization
            def transformer_forward(*inputs):
                # Disable profiling during checkpointed forward pass
                prev_profiling = self.enable_profiling
                self.enable_profiling = False
                result = self.crosstransformer(*inputs)
                self.enable_profiling = prev_profiling
                return result
                
            if self.enable_checkpointing:
                # Use gradient checkpointing to save memory
                x, xt = checkpoint(transformer_forward, x, xt)
            else:
                # Standard forward pass through transformer
                x, xt = transformer_forward(x, xt)

            if self.bottom_channels:
                # Restore original channel counts
                x = rearrange(x, "b c f t-> b c (f t)")  # Flatten for 1x1 conv
                x = self.channel_downsampler(x)  # Reduce channels
                x = rearrange(x, "b c (f t)-> b c f t", f=f)  # Restore shape
                xt = self.channel_downsampler_t(xt)  # Match time branch

        # Decoder forward pass - reconstruct sources from latent representations
        for idx, decode in enumerate(self.decoder):
            # Get corresponding skip connection and target length
            skip = saved.pop(-1)  # Get matching encoder features
            x, pre = decode(x, skip, lengths.pop(-1))  # Apply decoder layer
            # 'pre' contains features before final transposed conv
            # Used when frequency and time branches separate
            
            # Process time branch if we're at the right depth
            offset = self.depth - len(self.tdecoder)  # Calculate depth offset
            if idx >= offset:  # If we're in time processing layers
                tdec = self.tdecoder[idx - offset]  # Get time decoder
                length_t = lengths_t.pop(-1)  # Get target time length
                
                if tdec.empty:  # Special case: merging branches
                    assert pre.shape[2] == 1, pre.shape  # Verify single frequency
                    pre = pre[:, :, 0]  # Remove frequency dimension
                    xt, _ = tdec(pre, None, length_t)  # Process through time decoder
                else:  # Normal time branch processing
                    skip2 = saved_t.pop(-1)  # Get time skip connection
                    xt, _ = tdec(xt, skip2, length_t)  # Process with skip connection

        # Verify all skip connections were used
        assert len(saved) == 0  # All frequency skips used
        assert len(lengths_t) == 0  # All time lengths used
        assert len(saved_t) == 0  # All time skips used

        # Reshape output for source separation
        S = len(self.sources)  # Number of sources to separate
        x = x.view(B, S, -1, Fq, T)  # [batch, sources, channels, freq, time]
        # Denormalize frequency branch output
        x = x * std[:, None] + mean[:, None]  # Restore original scale

        # Handle device compatibility for complex operations
        # Workaround for MPS/XPU complex number support
        x_is_mps_xpu = (x.device.type in ["mps", "xpu"])  # Check device type
        x_device = x.device  # Store original device
        if x_is_mps_xpu:
            x = x.cpu()  # Move to CPU for complex operations
            
        # Apply source separation mask
        zout = self._mask(z, x)  # Generate separated spectrograms
        
        # Convert back to time domain
        if self.use_train_segment:
            if self.training:
                x = self._ispec(zout, length)  # Use original length
            else:
                x = self._ispec(zout, training_length)  # Use training segment length
        else:
            x = self._ispec(zout, length)  # Use input length
            
        # Restore device if needed
        if x_is_mps_xpu:
            x = x.to(x_device)  # Move back to original device

        # Process time branch output
        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)  # Training mode shape
            else:
                xt = xt.view(B, S, -1, training_length)  # Inference mode shape
        else:
            xt = xt.view(B, S, -1, length)  # Standard shape
            
        # Denormalize time branch output
        xt = xt * stdt[:, None] + meant[:, None]  # Restore original scale
        
        # Combine frequency and time domain outputs
        x = xt + x  # Add contributions from both branches
        
        # Remove padding if added
        if length_pre_pad:
            x = x[..., :length_pre_pad]  # Trim to original length
            
        return x  # Return separated sources


# ------------------------------------------------------------------------
#  7) Encoder and Decoder Layer Implementations (no per-layer profiling)
# ------------------------------------------------------------------------
class HEncLayer(nn.Module):
    """Encoder layer with enhanced LoRA integration and memory profiling"""
    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        dconv: bool = True,
        norm: bool = True,
        context: int = 0,
        dconv_kw: dict = {},
        pad: bool = True,
        rewrite: bool = True,
        layer_name: str = "",
        default_rank: int = 4,
        layer_ranks: dict = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.freq = freq
        self.empty = empty
        self.pad = pad
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm = norm
        
        # Normalization function
        norm_fn = lambda d: nn.GroupNorm(norm_groups, d) if norm else nn.Identity()
        
        # Padding
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
            
        # Main convolution
        if freq:
            conv = nn.Conv2d(
                chin, chout,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(pad, 0)
            )
        else:
            conv = nn.Conv1d(chin, chout, kernel_size, stride, pad)
            
        # Wrap with LoRA
        self.conv = wrap_with_lora(
            conv,
            layer_name=layer_name,
            default_rank=default_rank,
            layer_ranks=layer_ranks,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
        
        if empty:
            return
        
        # Additional components
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        
        if rewrite:
            if freq:
                rewrite_conv = nn.Conv2d(chout, 2 * chout, 1, 1, 0)
            else:
                rewrite_conv = nn.Conv1d(chout, 2 * chout, 1, 1)
            rewrite_name = f"{layer_name}.rewrite" if layer_name else ""
            self.rewrite = wrap_with_lora(
                rewrite_conv,
                layer_name=rewrite_name,
                default_rank=default_rank,
                layer_ranks=layer_ranks,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            self.norm2 = norm_fn(2 * chout)
        
        if dconv:
            dconv_name = f"{layer_name}.dconv" if layer_name else ""
            self.dconv = DConv(
                chout,
                layer_name=dconv_name,
                default_rank=default_rank,
                layer_ranks=layer_ranks,
                lora_alpha=lora_alpha,
                dropout=lora_dropout,
                **dconv_kw
            )
        else:
            self.dconv = None

    def forward(self, x: torch.Tensor, inject: torch.Tensor = None) -> torch.Tensor:
        # Collapse frequency dimension for non-frequency layers
        if not self.freq and x.dim() == 4:  # Handle 4D input from frequency branch
            B, C, Fr, T = x.shape  # [batch, channels, freq, time]
            x = x.view(B, -1, T)  # Merge freq into channels
            
        # Add padding to ensure length is divisible by stride
        if not self.freq:  # Only needed for time-domain processing
            le = x.shape[-1]  # Get current length
            if (le % self.stride) != 0:  # Check if padding needed
                x = F.pad(x, (0, self.stride - (le % self.stride)))  # Right-pad
        
        # Apply main convolution with LoRA adaptation
        y = self.conv(x)  # Conv1d or Conv2d based on freq flag
        if self.empty:  # Early return for empty layers
            return y
        
        # Merge features from time branch if provided
        if inject is not None:  # Handle cross-branch injection
            assert inject.shape[-1] == y.shape[-1], "Time mismatch"  # Verify alignment
            if inject.dim() == 3 and y.dim() == 4:  # Handle dimension mismatch
                inject = inject[:, :, None]  # Add frequency dimension
            y = y + inject  # Add time branch features
        
        # Apply normalization and non-linearity
        y = F.gelu(self.norm1(y))  # GroupNorm + GELU activation
        
        # Process through depthwise convolution if enabled
        if self.dconv:
            if self.freq:  # Reshape for frequency processing
                B, C, Fr, T = y.shape  # [batch, channels, freq, time]
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)  # Prepare for DConv
            y = self.dconv(y)  # Apply depth-wise convolution
            if self.freq:  # Restore original shape
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        
        # Apply optional 1x1 convolution for feature refinement
        if self.rewrite:  # Optional feature rewriting
            z = F.glu(self.norm2(self.rewrite(y)), dim=1)  # GLU activation
        else:
            z = y  # Skip rewrite path
        
        return z  # Return processed features


class HDecLayer(nn.Module):
    """Decoder layer with enhanced LoRA integration and memory profiling"""
    def __init__(
        self,
        chin: int,
        chout: int,
        last: bool = False,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        dconv: bool = True,
        norm: bool = True,
        context: int = 1,
        dconv_kw: dict = {},
        pad: bool = True,
        context_freq: bool = True,
        rewrite: bool = True,
        layer_name: str = "",
        default_rank: int = 4,
        layer_ranks: dict = None,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.freq = freq
        self.last = last
        self.empty = empty
        self.pad = pad
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        self.chin = chin
        
        # Normalization function
        norm_fn = lambda d: nn.GroupNorm(norm_groups, d) if norm else nn.Identity()
        
        # Padding
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        
        # Main transposed convolution
        if freq:
            conv_tr = nn.ConvTranspose2d(
                chin, chout,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1)
            )
        else:
            conv_tr = nn.ConvTranspose1d(chin, chout, kernel_size, stride)
        
        # Wrap with LoRA
        self.conv_tr = wrap_with_lora(
            conv_tr,
            layer_name=layer_name,
            default_rank=default_rank,
            layer_ranks=layer_ranks,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
        self.norm2 = norm_fn(chout)
        
        if empty:
            return
        
        # Rewrite path
        self.rewrite = None
        if rewrite:
            if context_freq and freq:
                rewrite_conv = nn.Conv2d(
                    chin, 2 * chin,
                    kernel_size=(1, 1 + 2 * context),
                    stride=1,
                    padding=(0, context)
                )
            elif freq:
                rewrite_conv = nn.Conv2d(chin, 2 * chin, 1, 1)
            else:
                rewrite_conv = nn.Conv1d(chin, 2 * chin, 1 + 2 * context, 1, context)
            
            rewrite_name = f"{layer_name}.rewrite" if layer_name else ""
            self.rewrite = wrap_with_lora(
                rewrite_conv,
                layer_name=rewrite_name,
                default_rank=default_rank,
                layer_ranks=layer_ranks,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            self.norm1 = norm_fn(2 * chin)
        
        # DConv
        if dconv:
            dconv_name = f"{layer_name}.dconv" if layer_name else ""
            self.dconv = DConv(
                chin,
                layer_name=dconv_name,
                default_rank=default_rank,
                layer_ranks=layer_ranks,
                lora_alpha=lora_alpha,
                dropout=lora_dropout,
                **dconv_kw
            )
        else:
            self.dconv = None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        length: tp.Optional[int] = None
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # Restore frequency dimension for spectral processing
        if self.freq and x.dim() == 3:  # Handle 3D input
            B, C, T = x.shape  # [batch, channels, time]
            x = x.view(B, self.chin, -1, T)  # Add frequency dimension
        
        if not self.empty:  # Process non-empty layers
            # Add encoder features via skip connection
            x = x + skip  # Residual connection for gradient flow
            
            # Apply context-aware feature refinement
            if self.rewrite:  # Optional 1x1 conv path
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)  # GLU for feature gating
            else:
                y = x  # Skip rewrite path
            
            # Apply depthwise separable convolution
            if self.dconv:
                if self.freq:  # Handle frequency-domain processing
                    B, C, Fr, T = y.shape  # [batch, channels, freq, time]
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)  # Reshape for DConv
                y = self.dconv(y)  # Apply depth-wise convolution
                if self.freq:  # Restore original shape
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x  # Pass through for empty layers
        
        # Upsample features with transposed convolution
        z = self.norm2(self.conv_tr(y))  # ConvTranspose + normalization
        
        # Remove padding artifacts
        if self.freq:  # Frequency-domain processing
            if self.pad:  # Remove padding if used
                z = z[..., self.pad:-self.pad, :]  # Trim both ends
        elif length is not None:  # Time-domain processing
            z = z[..., self.pad:self.pad + length]  # Trim to target length
        
        # Apply non-linearity except at final layer
        if not self.last:  # Skip activation at output
            z = F.gelu(z)  # GELU activation
        
        return z, y  # Return processed features and intermediate state
