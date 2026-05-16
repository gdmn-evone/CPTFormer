# CPTFormer

This is the official implementation of CPTFormer: **Detecting Compressed AI-Generated Images via Phase Spectrum Robustness**.

The training setting and dataset protocol follow [ManyiLee/Open-world-Deepfake-Detection-Network](https://github.com/ManyiLee/Open-world-Deepfake-Detection-Network/tree/master). The model is built on [openai/CLIP](https://github.com/openai/CLIP) and [czczup/ViT-Adapter](https://github.com/czczup/ViT-Adapter).

## Training

The distributed training entry point is:

```bash
torchrun --nproc_per_node=<num_gpus> --standalone trainDDP_Phase.py
```

Before training, update the corresponding YAML configuration file in [`configs`](configs), such as [`configs/CPT2.yaml`](configs/CPT2.yaml) or [`configs/CPT4.yaml`](configs/CPT4.yaml). In particular, set the dataset root, experiment name, checkpoint directory, train/validation split, and model path fields according to your local environment.

## Checkpoint

The 2-class model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/15ittNJHudGbX8dwNpZg_F34c4oHyZI0t?usp=drive_link).
