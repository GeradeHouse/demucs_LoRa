# Training HT-Demucs with LoRA Adaptation

This guide covers how to train HT-Demucs models using Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning method that enables rapid adaptation of the model to new domains or specific separation tasks. We'll use Tech-House music separation as an example use case.

## Table of Contents
- [Setup and Installation](#setup-and-installation)
- [Datasets](#datasets)
- [LoRA Configuration](#lora-configuration)
- [Handling Source Count Mismatch and Source Linkage](#handling-source-count-mismatch-and-source-linkage)
- [Training Process](#training-process)
- [Model Architecture](#model-architecture)
- [Best Practices](#best-practices)
- [Model Export and Evaluation](#model-export-and-evaluation)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

---

## Setup and Installation

### Setting up Python Environment

1. **Create a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv demucs_env

   # Activate on Windows
   demucs_env\Scripts\activate

   # Activate on Linux/Mac
   source demucs_env/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   # Install PyTorch with CUDA support (adjust CUDA version if needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Clone the Demucs_LoRa repository
   git clone https://github.com/GeradeHouse/demucs_LoRa.git
   cd demucs_LoRa

   # Install base requirements
   pip install -r requirements.txt

   # Install additional dependencies for LoRA
   pip install einops xformers
   ```

This setup ensures that you have both the standard Demucs environment plus the necessary packages for LoRA fine-tuning.

---

## Datasets

### Organizing Your Data

When performing source separation, each track in your dataset usually includes a `mixture.wav` file and separate stems for each source. Ensure each track’s folder is laid out like this:

```
dataset/
├── train/
│   ├── track1/
│   │   ├── mixture.wav
│   │   ├── other.wav
│   │   ├── kick.wav
│   │   ├── vocals.wav
│   │   ├── bass.wav
│   │   └── hihat.wav
│   ├── track2/
│   │   ├── mixture.wav
│   │   └── ...
│   └── ...
└── valid/
    ├── track1/
    │   ├── mixture.wav
    │   └── ...
    └── ...
```

1. **Alignment:** All stems must be perfectly time-aligned.  
2. **Integrity:** The sum of all stems should recreate the mixture without clipping.  
3. **Format:** Use uncompressed WAV files (16- or 24-bit) with a sample rate of 44.1kHz (or 48kHz).  

Having your data in this structure makes it easier for Demucs (and LoRA training) to locate and use the stems.

---

## LoRA Configuration

LoRA (Low-Rank Adaptation) allows you to fine-tune a large pretrained model by adding a small number of trainable parameters. This drastically reduces memory usage and training time compared to full fine-tuning.

Key LoRA parameters include:

- **lora_rank:** Base rank used for the low-rank adaptation matrices. (e.g., 4, 8, 16)
- **lora_alpha:** A scaling factor that multiplies the LoRA updates before adding them to the base weights.
- **lora_dropout:** Dropout applied to the LoRA layers for regularization.
- **lora_rank_mode:** How ranks are assigned to each layer (e.g., “uniform,” “heuristic,” “gradient”).

---

### Example Configuration (Split into Sections)

Below we split the **five_stem_lora.yaml** into smaller sections and explain each relevant block. This file is a sample configuration for training or fine-tuning a 5‑source HTDemucs model with LoRA.

#### 1. Defaults and Dataset

```yaml
#conf\variant\five_stem_lora.yaml
# @package _global_

defaults:
  - _self_
  - ../dset/musdb44
  - ../svd/default
```
- **defaults:** Points to other configuration files or defaults used alongside this one. `_self_` means “use the current file,” and the others reference additional configurations like musdb44 or default SVD settings.

```yaml
# Dataset configuration
dset:
  sources: ["other", "kick", "vocals", "bass", "hihat"]
  musdb_samplerate: 44100
  use_musdb: false
  wav: "./dataset"
  wav2:
  segment: 12
  shift: 1
  train_valid: false
  full_cv: true
  samplerate: 44100
  channels: 2
  normalize: true
  metadata: ./metadata
  valid_samples: null
```
- **sources:** Lists the stems you want to train for—here, we have 5 stems: *other*, *kick*, *vocals*, *bass*, and *hihat*.  
- **samplerate:** Must match your audio files.  
- **wav:** Directory containing your dataset (relative to this config).  
- **segment:** Length of audio segments (in seconds) used per training sample.  
- **use_musdb:** Set to `false` if you are providing a custom dataset.

#### 2. Model and Training Configuration

```yaml
# Model and training configuration
continue_from: "htdemucs"
continue_pretrained: null
continue_best: true
continue_opt: false
```
- **continue_from:** Name of the pretrained checkpoint or model you want to load. Here, "htdemucs" implies we start from a standard 4‑source model.  
- **continue_best:** If `true`, loads the best checkpoint rather than the last checkpoint.  
- **continue_opt:** If `true`, you’d also restore the previous optimizer state, but here it’s disabled.

```yaml
# Training settings
epochs: 360
batch_size: 16
max_batches: null
mixed_precision: true
```
- **epochs:** The number of total training epochs.  
- **batch_size:** Number of audio segments processed simultaneously on each training step.  
- **mixed_precision:** If `true`, uses half-precision to speed up training and reduce memory usage.

#### 3. Optimizer Settings

```yaml
optim:
  lr: 5e-5
  momentum: 0.9
  beta2: 0.999
  loss: l1
  optim: adam
  weight_decay: 0
  clip_grad: 0.5
  warmup_steps: 1000
  scheduler: cosine
```
- **lr:** Learning rate for the optimizer. `5e-5` is commonly used for fine-tuning.  
- **loss:** L1 loss is typical for audio source separation.  
- **clip_grad:** Enables gradient clipping at 0.5 to prevent overly large updates.  
- **warmup_steps:** Gradually increases the learning rate from 0 to `lr` during the first 1000 steps.  
- **scheduler:** Here, `"cosine"` gradually decreases the learning rate following a cosine curve.

#### 4. HTDemucs Model Configuration

```yaml
htdemucs:
  # Core architecture
  channels: 48
  channels_time: null
  growth: 2
  depth: 4
```
- **channels:** Base number of channels in the network (starting point).  
- **depth:** Number of layers in the encoder and decoder.

```yaml
  # STFT and processing parameters
  nfft: 4096
  wiener_iters: 0
  end_iters: 0
  wiener_residual: false
  cac: true
```
- **nfft:** Size of FFT used for frequency processing (e.g., 4096).  
- **cac (Complex-as-channels):** If `true`, treats real/imag parts as separate channels.  
- **wiener_iters:** Wiener filtering is disabled here (set to 0).

```yaml
  # Architecture features
  rewrite: true
  multi_freqs: []
  multi_freqs_depth: 3
  freq_emb: 0.2
  emb_scale: 10
  emb_smooth: true
```
- **rewrite:** Adds a 1×1 convolution to each layer to “rewrite” features.  
- **freq_emb:** Adds frequency embeddings with a weight factor of 0.2.  
- **emb_scale / emb_smooth:** Controls how embeddings are scaled and initialized.

```yaml
  # Convolution settings
  kernel_size: 8
  stride: 4
  time_stride: 2
  context: 1
  context_enc: 0
```
- **kernel_size:** The size of the convolution filter in each layer.  
- **stride:** How much to “hop” across the frequency dimension.  
- **time_stride:** Used in time-domain layers.  
- **context:** Additional context in the decoder with 1×1 convolutions.

```yaml
  # Normalization settings
  norm_starts: 4
  norm_groups: 4
```
- **norm_starts:** At which layer index normalization begins.  
- **norm_groups:** Sets how many groups to use for GroupNorm.

```yaml
  # DConv (Depthwise-Separable Convolution) settings
  dconv_mode: 1
  dconv_depth: 2
  dconv_comp: 8
  dconv_init: 1e-3
```
- **dconv_mode:** `1` means apply depthwise convolution in the encoder only.  
- **dconv_comp:** Compression factor in the DConv branch.  
- **dconv_init:** A scaling factor for initialization.

```yaml
  # Transformer settings
  bottom_channels: 0
  t_layers: 5
  t_hidden_scale: 4.0
  t_heads: 8
  t_dropout: 0.0
  t_layer_scale: true
  t_gelu: true
```
- **t_layers:** Number of transformer layers.  
- **t_heads:** Number of self-attention heads.  
- **t_gelu:** Whether to use GELU activation in the transformer blocks.

```yaml
  # Positional embedding settings
  t_emb: "sin"
  t_max_positions: 10000
  t_max_period: 10000.0
  t_weight_pos_embed: 1.0
  t_cape_mean_normalize: true
  t_cape_augment: true
  t_cape_glob_loc_scale: [5000.0, 1.0, 1.4]
  t_sin_random_shift: 0
```
- **t_emb:** Type of positional embedding; `"sin"` is a sinusoidal positional embedding.  
- **t_cape_mean_normalize:** Whether to normalize CAPE embeddings if used.

```yaml
  # Transformer normalization settings
  t_norm_in: true
  t_norm_in_group: false
  t_group_norm: false
  t_norm_first: true
  t_norm_out: true
  t_weight_decay: 0.0
  t_lr: null
```
- **t_norm_in:** Normalizes features before the attention blocks.  
- **t_lr:** Optionally specify a custom learning rate for only the transformer.

```yaml
  # Sparsity settings for transformer attention
  t_sparse_self_attn: false
  t_sparse_cross_attn: false
  t_mask_type: "diag"
  t_mask_random_seed: 42
  t_sparse_attn_window: 500
  t_global_window: 100
  t_sparsity: 0.95
  t_auto_sparsity: false
  t_cross_first: false
```
- **t_sparse_self_attn:** If `true`, uses local-sparse patterns for self attention (not used here).  
- **t_mask_type:** `"diag"` means we mask out attention outside a diagonal band.

```yaml
  # LoRA specific settings
  lora_rank: 8
  lora_alpha: 1.0
  lora_dropout: 0.1
```
- **lora_rank:** Base rank for LoRA. Higher rank = more capacity.  
- **lora_alpha:** Scales LoRA updates.  
- **lora_dropout:** Dropout used specifically within LoRA modules.

```yaml
  # Source linkage settings:
  source_linkage:
    other: other
    drums: kick
    vocals: vocals
    bass: bass
    hihat: new
```
- **source_linkage:** Tells the model how to map old sources to new ones if you’re continuing from a model with fewer (or different) sources.  
- Marking `"hihat"` as `"new"` indicates it has no corresponding pretrained weights.

```yaml
  layer_ranks:
    "encoder.0.*": 16
    "encoder.1.*": 12
    "transformer.*": 12
    "decoder.0.*": 12
    "decoder.1.*": 8
    "decoder.2.*": 6
  lora_rank_mode: "heuristic"
```
- **layer_ranks:** Overrides rank for certain layers (like the first encoder).  
- **lora_rank_mode:** `"heuristic"` uses a strategy where “important” layers get higher ranks.

```yaml
  # Memory optimization
  enable_checkpointing: true
  enable_profiling: false

  # Weight initialization
  rescale: 0.1
```
- **enable_checkpointing:** Uses gradient checkpointing to reduce memory usage.  
- **rescale:** A factor to rescale weights after initialization to keep them stable.

#### 5. Additional Training Settings

```yaml
training:
  batch_size: 1
  gradient_clip: 0.5
  mixed_precision: true
  epochs: 360
  optim:
    lr: 5e-5
    warmup_steps: 1000
    scheduler: cosine
```
- **training:** You can override certain parameters here—like `batch_size`—without editing them inside the main `htdemucs` block.

```yaml
# SVD settings for LoRA
svd:
  penalty: 0
  ...
```
- These advanced settings handle certain aspects of low-rank factorization or SVD if needed.

```yaml
# Quantization settings
quant:
  diffq: null
  ...
```
- Typically not used during training, but provided if you wish to experiment with quantization.

```yaml
# Output and logging
dora:
  dir: outputs
  exclude: ["misc.*", "slurm.*", "test.reval", "flag", "dset.backend"]
```
- **dir:** Where all logs, checkpoints, and generated audio examples are saved.

```yaml
# SLURM cluster usage (optional)
slurm:
  time: 4320
  constraint: volta32gb
  setup: ['module load ...']
```
- If training on a SLURM-based cluster, you can specify job constraints here.

```yaml
# Logging config
hydra:
  job_logging:
    formatters:
      colorlog:
        datefmt: "%m-%d %H:%M:%S"
```
- Adjust the date/time format for logs if needed.

---

## Handling Source Count Mismatch and Source Linkage

**Why do we need source linkage?** Sometimes you have a pretrained model with a certain set of sources, e.g., `["drums", "bass", "vocals", "other"]`, but now want to fine-tune it on a dataset that has either more or fewer sources. The **source_linkage** block tells the training code how to remap pretrained weights to your new stems.

- **Mark “new” sources** (like `"hihat"`) so the code knows to initialize them from scratch.
- For example, if your pretrained model had `"drums"` but your new dataset has both `"kick"` and `"hihat"`, you might map `"drums" → "kick"` and set `"hihat" : "new"`.

**Steps to ensure correct setup:**
1. **Update** `dset.sources` in your config to reflect the new stems.  
2. **Define** a `source_linkage` mapping in `htdemucs` that remaps old pretrained stems to the new ones.  
3. **Verify** that the final number of stems in your new model and dataset line up.

---

## Training Process

### Monitoring and Logging

Demucs logs are stored in the **`outputs/<signature>`** folder. Key logs include:
- **train.log:** Real-time updates on loss and iteration count.
- **validation/ folder:** Validation SDR results (if computed).
- **checkpoints/ folder:** Periodic model checkpoints.

**Example tips:**
1. **View live logs** in your terminal:
   ```bash
   tail -f outputs/<experiment_signature>/logs/train.log
   ```
2. **Check GPU usage**:
   ```bash
   nvidia-smi -l 1
   ```
3. **Launch TensorBoard** to visualize metrics:
   ```bash
   tensorboard --logdir outputs/<experiment_signature>/tensorboard
   ```

4. **Key Metrics and their Location:**
    THe loss curves is a good indicator of training progress. You can find them in the `outputs/<signature>/tensorboard/` folder.
    Model checkpoints are the saved weights of the model at different training stages. They are stored in the `outputs/<signature>/checkpoints/` folder.
    Validation results are the SDR metrics computed on the validation set. They are stored in the `outputs/<signature>/validation/` folder.
    Configuration files are stored in the `outputs/<signature>/config.yaml` file.


### Starting from Scratch

To create a brand new model with LoRA:

```bash
dora run -d model=htdemucs \
  htdemucs.lora_rank=4 \
  htdemucs.lora_alpha=1.0 \
  htdemucs.channels=48
```

**Explanation:**  
- **`model=htdemucs`**: We choose the HTDemucs base architecture.  
- **`lora_rank=4`**: A modest rank that saves memory while still enabling adaptation.  
- **`channels=48`**: The base number of channels in the network.

### Fine-tuning Existing Models

1. **Fine-tuning a 4‑source model** (e.g., `htdemucs`):
   ```bash
   dora run -d -f 955717e8 \
     continue_from=955717e8 \
     htdemucs.lora_rank=4 \
     variant=finetune
   ```
   - **`-f 955717e8`** references the experiment signature you want to start from.
   - **`variant=finetune`** is just a name for your experiment run.

2. **Fine-tuning a 6‑source model** (e.g., `htdemucs_6s`):
   ```bash
   dora run -d -f htdemucs_6s \
     continue_from=htdemucs_6s \
     htdemucs.lora_rank=8 \
     variant=finetune
   ```
   - Similar approach, but now we specify `lora_rank=8` for deeper adaptation.

### Training Command for 5‑Source Model

Below is a **human-readable explanation** for an example command:

```bash
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

1. **`-d`**: Runs in detached mode, enabling parallel logging.  
2. **`-f htdemucs`**: Base experiment name, meaning we start from a reference named `"htdemucs"`.  
3. **`continue_from=htdemucs`**: Tells the script to load from a pretrained “htdemucs” checkpoint.  
4. **`htdemucs.lora_rank=8`**: We are using a rank of 8 for LoRA, giving more capacity than rank=4.  
5. **`variant=finetune`**: Labels this training session as a fine-tuning variant in the logs.  
6. **`dset.sources='["other", "kick", "vocal", "bass", "hihat"]'`**: The new dataset has these 5 sources.  
7. **`dset.wav="path/to/5stem_dataset"`**: Points to your custom 5-stem dataset.  
8. **`optim.lr=5e-5`**: Low learning rate, typical for fine-tuning.  
9. **`optim.warmup_steps=500`**: We warm up the learning rate over 500 steps.  
10. **`htdemucs.nfft=4096`**: Use a 4096 FFT size for better frequency resolution.  
11. **`htdemucs.segment=12`**: Segment length is 12 seconds, giving the model a good chunk of audio to learn from.

---

## Model Architecture

## Model Architecture

HTDemucs combines multiple processing methods to capture diverse musical features:

1. **Dual-Path Processing**  
  - **Frequency Domain (STFT-based):** Analyzes harmonic and tonal components by converting waveforms into a frequency representation.  
  - **Time Domain (Raw Waveforms):** Detects transients, rhythms, and other time-specific details directly from raw audio.  
  - **Fusion:** Both paths are blended at crucial network stages, so the model benefits from frequency clarity and time-domain precision.

2. **Transformer Integration**  
  - **Cross-Attention:** The model’s time and frequency branches can exchange information, enhancing separation of intricate elements like vocals or percussion.  
  - **Positional Embeddings:** These help align audio features over time, ensuring the model tracks temporal context.  
  - **Sparse Attention (Optional):** Cuts down computations for long audio signals by limiting attention computations to relevant regions.

3. **LoRA Adaptation**  
  - **Lightweight Fine-Tuning:** Only small, low-rank layers are trained, while the main model stays frozen to preserve its general knowledge.  
  - **Speed and Efficiency:** LoRA drastically reduces the number of trainable parameters, making adaptation to new music genres or instrument tracks faster.  
  - **Domain Specialization:** Perfect for tasks like adding niche stems (e.g., new instruments) or retraining in a custom audio domain.


## Best Practices

### Rank Selection

Your LoRA rank can significantly affect memory usage and training quality. Some guidelines:

- **Small models** (≤48 channels): Ranks of 2–4 often suffice.  
- **Medium models** (64–128 channels): Ranks of 4–8.  
- **Large models** (≥256 channels): Ranks of 8–16.  

### Learning Rate Guidelines

You typically want a smaller learning rate when fine-tuning versus training from scratch:

```yaml
# Example: two scenarios
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

**Explanation:**  
- **From scratch:** A higher LR (like 1e-4) can help the model learn more quickly since it starts from zero.  
- **Fine-tuning:** A lower LR (like 5e-5) is better to avoid overwriting the pretrained knowledge too aggressively.  
- **Warmup:** Gradually increases LR from 0 to the target over a certain number of steps. This helps stable convergence.  
- **Scheduler:** “Cosine” or “linear” annealing shapes how LR decreases over epochs.

### Memory Optimization

1. **Gradient Checkpointing:**
   ```yaml
   htdemucs.enable_checkpointing: true
   ```
   Helps reduce VRAM usage at the cost of extra compute.

2. **Mixed Precision:**
   ```yaml
   htdemucs.mixed_precision: true
   ```
   Speeds up training and reduces memory usage by using half-precision floats.

3. **Batch Size:**
   If you encounter OOM (Out Of Memory) errors, try reducing your batch size from 16 to 8 or even 1.

---

## Model Export and Evaluation

### Exporting Models

When your training finishes, you can export the resulting model:

```bash
# Export single model
python3 -m tools.export <signature>

# Export with LoRA weights
python3 -m tools.export <signature> --include-lora
```
- **`<signature>`**: The ID or name referencing your experiment run.

### Model Evaluation

```bash
# Evaluate on test set
python3 -m tools.test_pretrained -n <signature>

# For a custom evaluation with multiple overlap shifts
python3 -m tools.test_pretrained -n <signature> test.shifts=2
```
- The script automatically loads the best checkpoint from your specified signature and evaluates it on your test data.

### Using Trained Models

You can load your trained (and possibly LoRA-fine-tuned) model in Python:

```python
from demucs import pretrained
from demucs.apply import apply_model

# Load model with LoRA
model = pretrained.get_model('your_model_signature')
model.load_lora_weights('path/to/lora_weights.pth')

# Process an audio mixture
sources = apply_model(model, mix)
```
- **`apply_model`** uses Demucs to separate the mixture into stems.  

### Quantization for Deployment

For efficient real-time usage:
```python
model.quantize(bits=8, optimize_scales=True)
torch.save({
    'state_dict': model.state_dict(),
    'lora_state': model.get_lora_state(),
    'config': model.config
}, 'quantized_model.pth')
```
- Reduces the model size at the cost of some fidelity.

---

## Advanced Configuration

### Layer-Specific Settings

LoRA parameters can be specialized on a per-layer basis:

```yaml
htdemucs:
  layer_ranks:
    "encoder.0.*": 16    # early layers with more capacity
    "encoder.1.*": 12
    "transformer.*": 12
    "decoder.0.*": 12
    "decoder.1.*": 8
    "decoder.2.*": 6
```
- This approach invests “rank budget” where it’s most beneficial.

### Memory Profiling

To track memory usage:

```yaml
htdemucs:
  enable_profiling: true
```

### Custom Training Segments

```yaml
segment: 10
use_train_segment: true
```
- Forces consistent segment length (e.g., 10 s) in both training and inference, which can help reduce variability.

### Dynamic Source Count Handling

If your new dataset has a different number of sources than the pretrained model, ensure:
1. **`dset.sources`** is updated.  
2. **`source_linkage`** is defined to indicate how old sources are remapped.  
3. Newly introduced stems are marked as `new`.

---

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM):**
   ```yaml
   htdemucs.enable_checkpointing: true
   batch_size: <reduced_batch_size>
   htdemucs.mixed_precision: true
   ```
   Either reduce the batch size or enable checkpointing/mixed precision.

2. **Slow Convergence:**
   ```yaml
   optim.lr: 2e-4
   optim.warmup_steps: 2000
   ```
   Increase LR slightly and warm up for more steps if the loss is stagnating.

3. **Training Instability:**
   ```yaml
   htdemucs.lora_alpha: 1.0
   optim.clip_grad: 1.0
   ```
   Clipping gradients or lowering LoRA alpha can stabilize training.

### Validation

Watch the **loss curves** and **SDR metrics**:
- **Loss** should steadily decrease.  
- **SDR** (Signal-to-Distortion Ratio) should improve over time. Aim for an SDR above 5–7 dB to see decent separation.

---

## Performance Optimization

### Mixed Precision Training

```yaml
htdemucs.mixed_precision: true
```
Saves GPU memory and speeds up training by using half precision.

### Efficient Data Loading

```yaml
dset:
  num_workers: 4
  prefetch_factor: 2
  pin_memory: true
```
Speeds up data loading by buffering additional batches.

---

## Model Export and Deployment

### Exporting Trained Models

```bash
python -m tools.export <signature> --include-lora
```
This bundles the entire model, including LoRA layers, into a single checkpoint.

### Using Trained Models

```python
from demucs import pretrained
from demucs.apply import apply_model

model = pretrained.get_model('your_model_signature')
model.load_lora_weights('path/to/lora_weights.pth')
sources = apply_model(model, mix)
```

### Quantization for Deployment

```python
model.quantize(bits=8, optimize_scales=True)
torch.save({
    'state_dict': model.state_dict(),
    'lora_state': model.get_lora_state(),
    'config': model.config
}, 'quantized_model.pth')
```
Useful for on-device or real-time separation with minimal overhead.

---

## Advanced Topics

### Custom LoRA Architectures

If you want to customize how ranks scale with depth or implement novel LoRA strategies:
```python
class CustomLoRAConfig:
    def __init__(self, base_rank=4, rank_growth=1.5):
        self.base_rank = base_rank
        self.rank_growth = rank_growth
    
    def get_rank(self, layer_depth):
        return int(self.base_rank * (self.rank_growth ** layer_depth))
```

### Gradient-Based Rank Adaptation

An experimental mode that adjusts ranks dynamically based on gradient signals:
```yaml
htdemucs.lora_rank_mode: gradient
htdemucs.lora_rank_adaptation: true
htdemucs.min_rank: 2
htdemucs.max_rank: 16
```

---

## Final Remarks

We hope this documentation helps you fine-tune HT-Demucs with LoRA for new music genres or source sets. Pay special attention to:

- **LoRA rank** and **learning rate** adjustments if your domain differs from the original pretrained model.
- The **Handling Source Count Mismatch** section if you have a different number of stems than the pretrained model expects.
- **Memory optimization** (checkpointing, mixed precision) if you run out of GPU memory.

Happy training!