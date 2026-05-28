# DBFSNet

DBFSNet is a deepfake detection repository for training, evaluating, and inferring on face video data using a dual-branch spatial/frequency network with adaptive fusion.

---

# Overview

* `configs/config.yaml`: primary configuration file with dataset paths, model settings, training options, and output folders.
* `extract_faces.py`: extracts face crops from raw videos into `data/extracted_faces/`.
* `train.py`: trains the DBFSNet model and logs epoch metrics to CSV.
* `evaluate.py`: runs evaluation and generates visual/performance metrics.
* `infer.py`: performs inference for a single input image or video.
* `app.py`: Streamlit frontend for interactive deepfake detection.
* `src/`: project source modules.

---

# Key Idea: Dynamic Adaptive Fusion

DBFSNet combines:

* **Spatial Features** → visual inconsistencies, blending artifacts, texture anomalies
* **Frequency Features** → hidden compression traces and frequency-domain manipulation patterns

Instead of treating both branches equally, DBFSNet uses **Adaptive Dynamic Fusion**.

### Static Fusion (Traditional Approach)

Earlier models often combine both feature branches with fixed importance.

Example:

* Image A may contain stronger frequency artifacts
* Image B may contain stronger spatial inconsistencies

Yet static fusion treats both patterns similarly.

---

### Adaptive Fusion (DBFSNet Approach)

DBFSNet dynamically adjusts feature importance for every input image.

Example:

* Image A → frequency branch receives higher importance
* Image B → spatial branch receives higher importance

This allows the model to better generalize across different deepfake generation techniques and compression levels.

### Fusion Equation

```text
Fused = g ⊙ Spatial + (1 - g) ⊙ Frequency
```

Where:

* `g → 1` : trust spatial features more
* `g → 0` : trust frequency features more
* `g → 0.5` : balanced fusion

The adaptive gate learns which feature domain is more reliable for each sample dynamically during training.

---

# Dataset Information

This project uses the **FaceForensics++** dataset for deepfake detection research.

## Compression Level Used

* **C40 Compression**
* Chosen to simulate realistic social media/video platform compression conditions.

---

## Number of Videos Used in FaceForensics++ (C40)

| Category                   | Number of Videos |
| -------------------------- | ---------------- |
| Original Real Videos       | 1,000            |
| DeepFakes Manipulated      | 1,000            |
| Face2Face Manipulated      | 1,000            |
| FaceSwap Manipulated       | 1,000            |
| NeuralTextures Manipulated | 1,000            |

---

## Frame Extraction

Instead of directly training on full videos, frames were extracted from videos using:

```bash
python extract_faces.py
```

The extracted face frames are then used for training and evaluation.

---

# Requirements

Install dependencies in a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

---

# Configuration

Update `configs/config.yaml` before running the pipeline.

## Key Config Sections

* `data.real_video_dir`: path to real videos
* `data.fake_video_dirs`: paths to fake/manipulated videos
* `data.extracted_faces_dir`: path to store face crops
* `training`: epochs, batch size, learning rate, scheduler, etc.
* `paths.checkpoint_dir`: directory for model checkpoints
* `paths.output_dir`: directory for saved outputs
* `paths.log_dir`: directory for logs
* `paths.metrics_csv`: CSV file for per-epoch training metrics

The current config uses:

```text
C:/Users/gauri/Desktop/DBFS
```

Update paths if your repository is moved.

---

# Typical Workflow

## Extract Faces

```bash
python extract_faces.py
```

---

## Train the Model

```bash
python train.py
```

---

## Evaluate the Model

```bash
python evaluate.py
```

---

## Run Inference

```bash
python infer.py --input "path/to/image_or_video"
```

---

## Launch Streamlit App

```bash
streamlit run app.py
```

---

# Repository Structure

```text
.
├── app.py
├── configs/config.yaml
├── data/
├── extract_faces.py
├── infer.py
├── outputs/
├── requirements.txt
├── train.py
├── src/
└── .gitignore
```

---

# Notes

* Keep `num_workers = 0` on Windows to avoid worker spawning issues.
* The `data/`, `outputs/`, and `checkpoints/` directories are excluded from version control via `.gitignore`.
* If config paths are changed, verify:

  * `checkpoint_dir`
  * `output_dir`
  * `metrics_csv`

---

# GitHub Preparation

This repository excludes generated artifacts and local dataset files from Git using `.gitignore`.

Only source code, configuration files, and documentation are committed.

---

# Project Objective

The goal of DBFSNet is to improve deepfake detection robustness under:

* Heavy compression
* Multiple manipulation techniques
* Real-world social media degradation
* Frequency-domain artifact suppression

The adaptive spatial-frequency fusion mechanism helps the model dynamically focus on the most informative feature representation for each sample.
