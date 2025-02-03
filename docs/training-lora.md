# Training HT-Demucs with LoRA Adaptation

This guide covers how to train HT-Demucs models using Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning method that enables rapid adaptation of the model to new domains or specific separation tasks. We'll use Tech-House music separation as an example use case.

## Table of Contents
- [Setup and Installation](#setup-and-installation)
- [Datasets](#datasets)
- [LoRA Configuration](#lora-configuration)
- [Training Process](#training-process)
- [Model Architecture](#model-architecture)
- [Best Practices](#best-practices)
- [Model Export and Evaluation](#model-export-and-evaluation)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

## Setup and Installation

### Setting up Python Environment

1. Create a virtual environment:
```bash
# Create virtual environment
python -m venv demucs_env

# Activate on Windows
demucs_env\Scripts\activate

# Activate on Linux/Mac
source demucs_env/bin/activate
```

2. Install dependencies:
```bash
# Install PyTorch with CUDA support (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone Demucs repository
git clone https://github.com/GeradeHouse/demucs_LoRa.git
cd demucs

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for LoRA
pip install einops xformers
```

## Understanding HTDemucs Architecture

HTDemucs combines three key components:

1. **Dual-Path Processing**
   - Frequency domain: Processes spectrograms through frequency-domain convolutions
     * STFT-based analysis for frequency patterns
     * Specialized handling of bass and percussion
   - Time domain: Parallel processing of raw waveforms
     * Direct waveform analysis for temporal features
     * Preserves phase information
   - Cross-path fusion: Merges information between paths at strategic points
     * Combines spectral and temporal features
     * Maintains coherence between domains

2. **Transformer Integration**
   - Cross-attention between time and frequency representations
     * Learns long-term dependencies
     * Captures musical structure
   - Position-aware processing through various embedding options
     * Handles varying song lengths
     * Maintains temporal context
   - Optional sparse attention patterns for efficiency
     * Reduces memory usage
     * Focuses on relevant time-frequency relationships

3. **LoRA Adaptation**
   - Memory-efficient parameter updates through low-rank decomposition
     * Reduces training parameters by 95%+
     * Enables rapid fine-tuning
   - Selective freezing of base model weights
     * Preserves general separation capabilities
     * Allows focused genre adaptation
   - Layer-specific rank allocation for optimal adaptation
     * Higher ranks for critical layers
     * Efficient parameter distribution

## Genre-Specific Training

## Extended Source Separation Configuration

For 5-source separation with specialized drum components:

```yaml
dset:
  sources: ["other", "kick", "vocals", "bass", "hihat"]  # 5-source configuration
  sample_rate: 44100  # Match your audio files
  channels: 2  # Stereo processing
  
htdemucs:
  # Enhanced LoRA settings for specialized drum separation
  lora_rank_mode: heuristic
  lora_rank: 8  # Higher base rank for detailed separation
  
  # Layer-specific ranks optimized for drum component isolation
  layer_ranks:
    # Early layers need higher ranks to learn detailed features
    "encoder.0.*": 16      # Critical for initial feature extraction
    "encoder.1.*": 12      # Focus on rhythmic and timbral patterns
    "transformer.*": 12    # Attention to temporal relationships
    "decoder.0.*": 12     # Detailed reconstruction
    "decoder.1.*": 8      # Refined separation
    "decoder.2.*": 6      # Final adjustments
  
  # Audio processing optimized for percussion
  nfft: 4096              # Larger FFT for better frequency resolution
  segment: 12             # Longer segments for pattern recognition
  hop_length: 1024        # Overlap for smoother transitions
  
  # Training settings
  batch_size: 32          # Adjust based on GPU memory
  gradient_clip: 0.5      # Prevent gradient explosions
  mixed_precision: true   # Improve training speed
```

### Training Command for 5-Source Model

```bash
# Start from pre-trained 4-source model and adapt to 5 sources with specialized drum separation
dora run -d -f htdemucs \
  continue_from=htdemucs \
  htdemucs.lora_rank=8 \
  variant=finetune \
  dset.sources='["other", "kick", "vocal", "bass", "hihat"]' \
  dset.wav="path/to/5stem_dataset" \
  optim.lr=5e-5 \
  optim.warmup_steps=500 \
  htdemucs.nfft=4096 \
  htdemucs.segment=12
```

### Dataset Requirements and Recommendations

1. **Dataset Size Recommendations**
   - Minimum Requirements:
     * Training set: 20-30 songs (2-3 hours of music)
     * Validation set: 5-10 songs (30-60 minutes)
   - Optimal Setup:
     * Training set: 50-100 songs (5-10 hours)
     * Validation set: 15-20 songs (1.5-2 hours)
   - Professional Setup:
     * Training set: 200+ songs (20+ hours)
     * Validation set: 40-50 songs (4-5 hours)

2. **Audio Quality Requirements**
   - Sample rate: 44.1kHz or 48kHz
   - Bit depth: 16-bit or 24-bit
   - Format: WAV (uncompressed)
   - Duration: Full tracks (3-7 minutes each)

### Dataset Preparation

1. **Organize Your Training Data**
   ```
   dataset/
   ├── train/                      # Training dataset directory
   │   ├── track1/                 # Each track in its own folder
   │   │   ├── mixture.wav        # Original mixed track
   │   │   ├── other.wav         # All instruments except drums/bass
   │   │   ├── kick.wav          # All drums except hi-hats
   │   │   ├── vocals.wav         # All vocal elements
   │   │   ├── bass.wav          # Bass instruments
   │   │   └── hihat.wav         # Hi-hat patterns only
   │   ├── track2/
   │   │   ├── mixture.wav
   │   │   └── ...
   │   └── ...
   └── valid/                      # Validation dataset directory
       ├── track1/                 # Same structure as training
       │   ├── mixture.wav
       │   └── ...
       └── ...
   ```

2. **Audio File Requirements**
   - Each stem must be perfectly aligned
   - Sum of stems should equal the mixture
   - No clipping or distortion
   - Consistent loudness levels

3. **Stem Quality Guidelines**
   - Other:
     * Clean separation from drums and bass
     * Well-preserved melodic content
     * Minimal rhythmic bleed-through
   - Kick:
     * All drum elements except hi-hats
     * Clean transients for all percussion
     * No hi-hat bleed in drum hits
   - Vocal:
     * Clear separation from instruments
     * All vocal processing included
     * No instrumental bleed
   - Bass:
     * Clear sub frequencies
     * Minimal mid-range bleed
     * Consistent phase alignment
   - Hi-hat:
     * Isolated hi-hat patterns only
     * Clean cymbal separation
     * No bleed from other drums

4. **Special Considerations for Drum Component Separation**
   - Ensure complete isolation of hi-hats from other drums
   - Maintain phase coherence between kick and hi-hat stems
   - Verify no hi-hat artifacts in kick stem
   - Check for proper transient preservation in both stems
   - Monitor frequency crossover points between components

### Configuration Setup

1. **Update Dataset Path**
   ```yaml
   # conf/config.yaml
   dset:
     wav: "path/to/tech_house_data"  # Absolute path to dataset
     sources: ["drums", "bass", "other", "vocals"]  # Source order
     sample_rate: 44100  # Match your audio files
     channels: 2  # Stereo processing
   ```

2. **Verify Dataset Loading**
   ```bash
   # Test dataset loading
   python -m tools.verify_dataset path/to/tech_house_data
   ```

### Dataset Metadata Cache

Datasets are scanned on first use. To force rescan:
```bash
rm -rf metadata/

## LoRA Configuration

### Basic Parameters

```yaml
htdemucs:
  # Core LoRA settings
  lora_rank: 4              # Default rank for adaptation matrices
  lora_alpha: 1.0           # Scaling factor for updates
  lora_dropout: 0.1         # Dropout for regularization
  lora_rank_mode: heuristic # Rank allocation strategy
  
  # Advanced settings
  layer_ranks:              # Custom ranks per layer
    "encoder.0.*": 16      # Higher rank for early layers
    "transformer.*": 12    # Medium rank for transformer
    "decoder.*": 4        # Lower rank for decoder
  lora_max_rank: 16        # Maximum allowed rank
```

### Rank Allocation Modes

1. Uniform Mode:
   - Consistent rank across all layers
   - Good baseline for initial experiments
   ```yaml
   htdemucs.lora_rank_mode: uniform
   htdemucs.lora_rank: 4
   ```

2. Heuristic Mode (Recommended):
   - Layer-specific ranks based on importance
   - Critical layers (encoder.0, transformer): 8-16
   - Middle layers: 4-8
   - Final layers: 2-4
   ```yaml
   htdemucs.lora_rank_mode: heuristic
   htdemucs.lora_rank: 4  # Base rank for scaling
   ```

3. Gradient Mode:
   - Dynamic rank adaptation (experimental)
   ```yaml
   htdemucs.lora_rank_mode: gradient
   htdemucs.min_rank: 2
   htdemucs.max_rank: 16
   ```

### Tech-House Specific Fine-tuning

1. **Start from Pre-trained Model**
   ```bash
   # Fine-tune HTDemucs for Tech-House
   dora run -d -f htdemucs \
     continue_from=htdemucs \
     htdemucs.lora_rank=8 \
     variant=finetune \
     dset.wav="path/to/tech_house_data" \
     optim.lr=1e-4 \
     optim.warmup_steps=1000
   ```

2. **Genre-Optimized Configuration**
   ```yaml
   htdemucs:
     # Audio Processing
     nfft: 4096                # Larger FFT for better bass resolution
     segment: 15               # Longer segments for pattern recognition
     hop_length: 1024         # Overlap for smoother transitions
     
     # LoRA settings
     lora_rank: 8             # Higher rank for genre-specific features
     lora_alpha: 0.8          # Balanced adaptation strength
     lora_dropout: 0.1        # Regularization for stability
     
     # Layer-specific ranks
     layer_ranks:
       "encoder.0.*": 16      # High rank for initial feature extraction
       "encoder.1.*": 12      # Focus on rhythm patterns
       "transformer.*": 12    # Attention to long-term structure
       "decoder.*": 8        # Preserve genre-specific details
     
     # Training settings
     batch_size: 32          # Adjust based on GPU memory
     gradient_clip: 0.5      # Prevent gradient explosions
     mixed_precision: true   # Improve training speed
   ```

## Training Process

### Monitoring and Logging

1. **Access Training Logs**
   ```bash
   # View live training progress
   tail -f outputs/<experiment_signature>/logs/train.log
   
   # Monitor GPU usage
   nvidia-smi -l 1
   
   # View TensorBoard metrics
   tensorboard --logdir outputs/<experiment_signature>/tensorboard
   ```

2. **Important Metrics Location**
   - Loss curves: `outputs/<signature>/tensorboard/`
   - Model checkpoints: `outputs/<signature>/checkpoints/`
   - Validation results: `outputs/<signature>/validation/`
   - Configuration: `outputs/<signature>/config.yaml`

3. **Key Metrics to Monitor**
   - Training loss: Should decrease steadily
   - Validation SDR: Should improve over time
     * Good: > 5 dB
     * Great: > 7 dB
     * Excellent: > 10 dB
   - Per-source metrics:
     * Drums: Focus on transient clarity
     * Bass: Monitor low-end separation
     * Other: Check effect preservation
     * Vocals: Verify clean isolation

4. **Automatic Checkpointing**
   - Best model saved at: `outputs/<signature>/checkpoints/best.th`
   - Regular checkpoints: `outputs/<signature>/checkpoints/checkpoint_*.th`
   - LoRA weights: `outputs/<signature>/checkpoints/lora_*.th`

### Starting from Scratch

Train a new model with LoRA:

```bash
dora run -d model=htdemucs \
  htdemucs.lora_rank=4 \
  htdemucs.lora_alpha=1.0 \
  htdemucs.channels=48
```

### Fine-tuning Existing Models

1. Fine-tune 4-stem model (htdemucs):
```bash
dora run -d -f 955717e8 continue_from=955717e8 \
  htdemucs.lora_rank=4 \
  variant=finetune
```

2. Fine-tune 6-stem model (htdemucs_6s):
```bash
dora run -d -f htdemucs_6s continue_from=htdemucs_6s \
  htdemucs.lora_rank=8 \
  variant=finetune
```

### Training Configuration

Key parameters in `conf/config.yaml`:
```yaml
# Model architecture
htdemucs:
  channels: 48            # Initial number of channels
  growth: 2              # Channel growth per layer
  depth: 4              # Number of encoder/decoder layers
  
  # Transformer settings
  t_layers: 5           # Transformer layers
  t_heads: 8           # Attention heads
  t_dropout: 0.0       # Transformer dropout
  
  # LoRA settings
  lora_rank: 4         # Default adaptation rank
  lora_alpha: 1.0      # Update scaling
  lora_dropout: 0.1    # LoRA-specific dropout
  
  # Memory optimization
  enable_checkpointing: true  # For large models
  enable_profiling: false    # Memory tracking
```

## Model Architecture

HTDemucs combines:
1. Dual-Path Processing:
   - Frequency domain (STFT spectrograms)
   - Time domain (raw waveforms)
   
2. Transformer Integration:
   - Cross-attention between domains
   - Position-aware processing
   
3. LoRA Adaptation:
   - All convolutional layers
   - Transformer attention layers
   - Skip connections

### Monitoring Training

Monitor training progress through logs and metrics:

```python
from demucs.train import main as train_main

# Custom training monitoring
metrics = {
    'train_loss': [],
    'valid_loss': [],
    'nsdr': []
}

# Training with metric collection
train_main(args, metrics_callback=lambda m: metrics[m.name].append(m.value))
```


## Best Practices

### Rank Selection

Choose ranks based on model size:
- Small models (channels ≤ 48): rank 2-4
- Medium models (64-128): rank 4-8
- Large models (≥ 256): rank 8-16
### Learning Rate Guidelines

```python
# Recommended learning rate ranges for different scenarios
lr_configs = {
    'from_scratch': {
        'initial_lr': 1e-4,
        'warmup_steps': 1000,
        'schedule': 'cosine'
    },
    'fine_tuning': {
        'initial_lr': 5e-5,
        'warmup_steps': 500,
        'schedule': 'linear'
    }
}
```

### Memory Optimization

1. Enable gradient checkpointing:
```yaml
htdemucs.enable_checkpointing: true
```

2. Use mixed precision:
```yaml
htdemucs.mixed_precision: true
```

3. Adjust batch size based on GPU memory:
```yaml
batch_size: 32  # Reduce if OOM errors occur
```

### Learning Rate Guidelines

```yaml
optim:
  # From scratch
  lr: 1e-4
  warmup_steps: 1000
  scheduler: cosine
  
  # Fine-tuning
  lr: 5e-5
  warmup_steps: 500
  scheduler: linear
```

## Model Export and Evaluation

### Exporting Models

```bash
# Export single model
python3 -m tools.export <signature>

# Export with LoRA weights
python3 -m tools.export <signature> --include-lora
```

### Model Evaluation

```bash
# Evaluate on test set
python3 -m tools.test_pretrained -n <signature>

# Custom evaluation
python3 -m tools.test_pretrained -n <signature> test.shifts=2
```

### Using Trained Models

```python
from demucs import pretrained
from demucs.apply import apply_model

# Load model with LoRA
model = pretrained.get_model('your_signature')
model.load_lora_weights('path/to/weights.pth')

# Process audio
sources = apply_model(model, mix)
```

## Advanced Configuration

### Layer-Specific Settings

Fine-grained control over LoRA adaptation:

```yaml
htdemucs:
  layer_ranks:
    # Encoder layers
    "encoder.0.*": 16    # First encoder (critical)
    "encoder.1.*": 12    # Second encoder
    
    # Transformer layers
    "transformer.self_attn.*": 8
    "transformer.cross_attn.*": 12
    
    # Decoder layers
    "decoder.0.*": 8     # First decoder
    "decoder.1.*": 6     # Second decoder
    "decoder.2.*": 4     # Final decoder
```

### Memory Profiling

Enable detailed memory tracking:

```yaml
htdemucs:
  enable_profiling: true
  profile_memory: true
```

### Custom Training Segments

Adjust training segment length:

```yaml
# In config.yaml
segment: 10  # Training segment length in seconds
use_train_segment: true  # Use fixed length for inference
```

## Troubleshooting

### Common Issues

1. Out of Memory (OOM):
```yaml
# Solutions
htdemucs.enable_checkpointing: true
batch_size: batch_size // 2
htdemucs.mixed_precision: true
```

2. Slow Convergence:
```yaml
# Adjust learning rate and warmup
optim.lr: 2e-4
optim.warmup_steps: 2000
```

3. Training Instability:
```yaml
# Stabilize training
htdemucs.lora_alpha: min(lora_alpha, dim // n_heads)
htdemucs.gradient_clip: 1.0
```

### Validation

Monitor training progress:
- Loss curves should decrease steadily
- SDR metrics improve over time
- Memory usage remains stable

## Troubleshooting

Common issues and solutions:

1. Out of Memory (OOM):
```python
# Reduce memory usage
htdemucs.enable_checkpointing = True
htdemucs.batch_size = htdemucs.batch_size // 2
```

2. Slow Convergence:
```python
# Adjust learning rate and warmup
optim.lr = 2e-4
optim.warmup_steps = 2000
```

3. Unstable Training:
```python
# Stabilize training
htdemucs.lora_alpha = min(htdemucs.lora_alpha, htdemucs.dim // htdemucs.n_heads)
htdemucs.gradient_clip = 1.0
```

## Performance Optimization

### Mixed Precision Training

Enable mixed precision for faster training:

```python
# In your training configuration
htdemucs.mixed_precision = True
htdemucs.precision = 'float16'
```

### Efficient Data Loading

Optimize data loading:

```python
# In your configuration
dset:
  num_workers: 4
  prefetch_factor: 2
  pin_memory: True
```

## Model Export and Deployment

### Exporting Trained Models

Export your trained LoRA model:

```bash
python -m tools.export <signature> --include-lora
```

### Using Trained Models

Load and use your trained model:

```python
from demucs import pretrained
from demucs.apply import apply_model

# Load model with LoRA weights
model = pretrained.get_model('your_model_signature')
model.load_lora_weights('path/to/lora_weights.pth')

# Process audio
sources = apply_model(model, mix)
```

### Quantization for Deployment

Quantize model for efficient deployment:

```python
# 8-bit quantization
model.quantize(bits=8, optimize_scales=True)

# Export quantized model
torch.save({
    'state_dict': model.state_dict(),
    'lora_state': model.get_lora_state(),
    'config': model.config
}, 'quantized_model.pth')
```

## Advanced Topics

### Custom LoRA Architectures

Implement custom LoRA configurations:

```python
class CustomLoRAConfig:
    def __init__(self, base_rank=4, rank_growth=1.5):
        self.base_rank = base_rank
        self.rank_growth = rank_growth
    
    def get_rank(self, layer_depth):
        return int(self.base_rank * (self.rank_growth ** layer_depth))
```

### Gradient-Based Rank Adaptation

Enable dynamic rank adaptation:

```python
htdemucs.lora_rank_mode = 'gradient'
htdemucs.lora_rank_adaptation = True
htdemucs.min_rank = 2
htdemucs.max_rank = 16
```

