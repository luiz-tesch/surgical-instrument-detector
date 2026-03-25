# Surgical Instrument Detector

YOLOv8 fine-tuned to detect surgical instruments in laparoscopic and robotic surgery videos.

## Motivation

Automatic instrument detection is an open problem in surgical simulation and skill assessment. Being able to identify which instruments are present and used correctly is a key step toward providing real-time automated feedback in surgical training environments.

## Datasets

| Dataset | Instruments | Annotation type | Access |
|---|---|---|---|
| **EndoVis 2017** | 6 robotic instruments | Segmentation masks → boxes | [Zenodo](https://zenodo.org/records/10527017) |
| **Cholec80** | 7 laparoscopic instruments | Presence labels (no boxes) | [CAMMA](http://camma.u-strasbg.fr/datasets) |

## Project Structure

```
surgical-instrument-detector/
├── configs/
│   ├── data_endovis.yaml       # EndoVis 2017 dataset config
│   ├── data_cholec80.yaml      # Cholec80 dataset config
│   └── train_config.yaml       # Training hyperparameters
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_training_results.ipynb
├── src/
│   ├── prepare_endovis.py      # EndoVis 2017 → YOLO format
│   ├── prepare_cholec80.py     # Cholec80 → YOLO format
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── predict.py              # Inference script
├── requirements.txt
└── setup_env.sh
```

## Setup

```bash
# Clone
git clone https://github.com/luiz-tesch/surgical-instrument-detector
cd surgical-instrument-detector

# Create venv and install dependencies (Git Bash on Windows)
bash setup_env.sh
source venv/Scripts/activate

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Workflow

### 1. Prepare Dataset

Download `endovis2017.zip` from [Zenodo](https://zenodo.org/records/10527017) and place it in the project root, then:

```bash
python src/prepare_endovis.py --zip endovis2017.zip --output data/endovis2017
```

This reads directly from the zip, converts segmentation masks to YOLO bounding boxes, and splits into train/val/test.

### 2. Explore

```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### 3. Train

```bash
python src/train.py --data configs/data_endovis.yaml --model yolov8n.pt --epochs 100 --batch 16
```

### 4. Evaluate

```bash
python src/evaluate.py --weights runs/train/surgical_yolov8/weights/best.pt \
    --data configs/data_endovis.yaml --split test
```

### 5. Predict

```bash
python src/predict.py --weights runs/train/surgical_yolov8/weights/best.pt \
    --source path/to/video.mp4
```

## Results

### YOLOv8n — EndoVis 2017 (test set, 100 epochs)

| Class | mAP50 | mAP50-95 |
|---|---|---|
| monopolar_curved_scissors | 0.977 | 0.934 |
| vessel_sealer | 0.969 | 0.890 |
| large_needle_driver | 0.866 | 0.711 |
| bipolar_forceps | 0.833 | 0.743 |
| prograsp_forceps | 0.088 | 0.064 |
| **all** | **0.747** | **0.668** |

> Work in progress — experimenting with larger models (yolov8s) to improve detection of visually similar instruments (bipolar/prograsp forceps).

## Hardware

- GPU: NVIDIA GTX 1660 Ti (6GB VRAM)
- CUDA: 12.6
- PyTorch: 2.x (cu124)

## Citation

If you use the EndoVis 2017 dataset, please cite:

```bibtex
@dataset{sun_2024_10527017,
  author    = {Sun, Liping},
  title     = {Endovis2017 \& Endovis2018 Datasets preprocessed for
               Contrastive Segmentation in Endoscopic robotic surgery},
  year      = {2024},
  doi       = {10.5281/zenodo.10527017},
  url       = {https://zenodo.org/records/10527017}
}

@article{Allan2019,
  title   = {2017 Robotic Instrument Segmentation Challenge},
  author  = {Allan, Max and Shvets, Alex and Kurmann, Thomas and Zhang, Zichen
             and Duggal, Rahul and Su, Yun-Hsuan and Rieke, Nicola and Laina, Iro
             and Kalavakonda, Niveditha and Bodenstedt, Sebastian and Herrera, Luis
             and Li, Wenqi and Iglovikov, Vladimir and Luo, Huoling and Yang, Jian
             and Stoyanov, Danail and Maier-Hein, Lena and Speidel, Stefanie
             and Azizian, Mahdi},
  journal = {arXiv preprint arXiv:1902.06426},
  year    = {2019}
}
```
