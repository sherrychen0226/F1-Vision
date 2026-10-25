# F1-Vision

Formula 1 car team & driver classification using deep learning — a **two-stage ResNet18 system** with a Gradio web demo: upload a car photo, and it first identifies the **team** (Ferrari, McLaren, Mercedes, Red Bull), then the **driver** within that team.

## Project Overview

This project builds an image classification system that recognizes F1 cars from racing screenshots and official photographs, across different viewing angles and track conditions.

The project tackles two tasks of increasing difficulty:

- **Team classification:** which of the four teams a car belongs to — largely solvable from livery colors and overall shape.
- **Driver classification:** which driver is in the car — much harder, because the distinguishing cue (the car number) often occupies only a small region of the image.

The project was developed in Python and PyTorch, with a Gradio interface for interactive image-based predictions.

## Repository Structure

| File | Purpose |
|---|---|
| `train_resnet18_finetune.py` | Fine-tunes ResNet18 on the 4-team dataset (`dataset/`, ImageFolder layout); saves the best-validation checkpoint |
| `train_driver_model.py` | Trains a team-specific binary driver classifier  |
| `app_v2.py` | Gradio web app chaining the two stages: team prediction → driver prediction with per-class probabilities |
| `predict_v12.py` | CLI tool for single-image team prediction with a full probability breakdown |

> Datasets and trained checkpoints (`checkpoints/*.pth`) are not included in the repo; see *Reproducing* below.

## How It Works

The system uses **hierarchical two-stage classification**:

```
Input image ──► Stage 1: Team model (ResNet18, 224×224)
                    │  Ferrari / McLaren / Mercedes / Red Bull
                    ▼
                Stage 2: Team-specific driver model (ResNet18, 384×384)
                    │  e.g. Ferrari → Hamilton vs. Leclerc
                    ▼
                Driver prediction (or "Coming Soon" for Red Bull)
```

Driver pairs currently supported: **Ferrari** (Hamilton / Leclerc), **McLaren** (Norris / Piastri), **Mercedes** (Antonelli / Russell). The Red Bull driver classifier is not yet trained.

## Methodology

- **Dataset:** ~750 images collected from racing screenshots, official car photographs, and other on-track shots, covering varied angles and conditions. 80/20 train/val split (fixed seed) for every experiment.
- **Transfer learning:** pretrained ResNet18 with all layers frozen except `layer4` and the classifier head — fine-tuning only the last residual block keeps training light on a small dataset.
- **Team model:** 224×224 input, Adam, 10 epochs, checkpointing the best validation accuracy.
- **Driver models:** separate binary classifiers per team, trained at **384×384** resolution to preserve small details like car numbers; no horizontal-flip augmentation, since flipping corrupts number-specific visual information.
- **Inference:** both stages run back-to-back inside one Gradio request, showing the predicted class, confidence, and full per-class probability distributions.

## Results

- **Team classification:** ~97% validation accuracy during fine-tuning.
- **Driver classification:** less consistent, up to ~85% validation accuracy in the best experiments; performance varied between teams and was limited by training data size and car-number visibility.

> These are validation results on the project's collected dataset, not on a fully independent external benchmark.

## Reproducing

1. Organize images as ImageFolder datasets:
   ```text
   dataset/Ferrari/ ... dataset/McLaren/ ... dataset/Mercedes/ ... dataset/RedBull/
   driver_dataset/Ferrari/Hamilton/ ... driver_dataset/Ferrari/Leclerc/ ...
   ```
2. Install dependencies: `torch`, `torchvision`, `gradio`, `Pillow`, `matplotlib`.
3. Train:
   ```bash
   python train_resnet18_finetune.py
   python train_driver_model.py 
  
   ```
4. Run the demo or the CLI:
   ```bash
   python app_v2.py        # launches the Gradio web interface
   python predict_v12.py path/to/image.jpg
   ```
