# DBFSNet

DBFSNet is a deepfake detection repository for training, evaluating, and inferring on face video data using a dual-branch spatial/frequency network.

## Overview

- `configs/config.yaml`: primary configuration file with dataset paths, model settings, training options, and output folders.
- `extract_faces.py`: extracts face crops from raw videos into `data/extracted_faces/`.
- `train.py`: trains the DBFSNet model and logs epoch metrics to CSV.
- `evaluate.py`: runs evaluation and generates visual/performance metrics.
- `infer.py`: performs inference for a single input image or video.
- `app.py`: Streamlit frontend for interactive deepfake detection.
- `src/`: project source modules.

## Requirements

Install dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Update `configs/config.yaml` before running the pipeline.

Key config sections:

- `data.real_video_dir`: path to real videos
- `data.fake_video_dirs`: paths to fake/manipulated videos
- `data.extracted_faces_dir`: path to store face crops
- `training`: epochs, batch size, learning rate, scheduler, etc.
- `paths.checkpoint_dir`: directory for model checkpoints
- `paths.output_dir`: directory for saved outputs
- `paths.log_dir`: directory for logs
- `paths.metrics_csv`: CSV file for per-epoch training metrics

> The current config uses `C:/Users/sanid/Desktop/DBFS`, so update paths if your repository is moved.

## Typical workflow

1. Extract faces:

```powershell
python extract_faces.py
```

2. Train the model:

```powershell
python train.py
```

3. Evaluate the model:

```powershell
python evaluate.py
```

4. Run inference on a file:

```powershell
python infer.py --input "path/to/image_or_video"
```

5. Launch the Streamlit app:

```powershell
streamlit run app.py
```

## Notes

- Keep `num_workers` set to `0` on Windows to avoid worker spawning issues.
- The `data/`, `outputs/`, and `checkpoints/` directories are excluded from version control via `.gitignore`.
- If you change config paths, verify the `checkpoint_dir`, `output_dir`, and `metrics_csv` locations.

## Repository structure

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

## GitHub preparation

This repo excludes generated artifacts and local dataset files from Git using `.gitignore`. Commit only source code, config files, and documentation.
