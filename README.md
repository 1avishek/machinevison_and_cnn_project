# Machine Vision and CNN Final Project

Semantic segmentation of femur X-rays using a PyTorch UNet implementation. The
project covers dataset preparation, training, evaluation, and inference.

## Project Structure

- `pth_fix.py` – rebuilds `data/csvs/{dataset,train,val,test}.csv` by scanning the
  X-ray and mask folders.
- `dataset.py` – `Knee_dataset` class with optional mask handling for unlabeled
  samples.
- `model.py` – configurable UNet encoder/decoder implementation.
- `trainer.py` – training loop (BCE + Dice losses) with checkpointing.
- `main.py` – entry point for training.
- `evaluate.py` – loads a saved checkpoint, reports metrics, and saves a Dice
  bar chart.
- `predict.py` – (WIP/optional) helper for running inference on unlabeled data.
- `data/` – raw images, masks, CSV splits, evaluation chart output.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install torch torchvision pandas opencv-python matplotlib
```

Run `python3 pth_fix.py` whenever the dataset contents change. This recreates the
CSV splits expected by the training/evaluation scripts.

## Training

```bash
python3 main.py --epochs 10 --batch_size 16 --lr 1e-3 --wd 1e-5 --out_dir session
```

Key outputs:

- Console log with per-epoch train/val losses and Dice.
- `session/best_model.pt` – checkpoint with highest validation Dice.
- `session/last_model.pt` – checkpoint from the final epoch.

Use `--batch_size {16,32,64}` as defined in `args.py`. Adjust `--csv_dir` if
your CSVs live elsewhere.

## Evaluation

Evaluate any labeled split (e.g., validation or dedicated test CSV) using the
best checkpoint:

```bash
python3 evaluate.py \
  --csv_path data/csvs/val.csv \
  --checkpoint session/best_model.pt \
  --batch_size 16 \
  --bar_path data/csvs/bar_chart.png
```

This reports average loss/Dice and writes a per-image Dice bar chart. Keep the
source visible and run this command during the video recording to satisfy the
assignment requirement.

## Inference / Predicted Masks

For unlabeled samples (rows without masks), instantiate the dataset with
`return_mask=False` and load the trained model:

```python
from dataset import Knee_dataset
from model import UNetLext
import pandas as pd
import torch

df = pd.read_csv('data/csvs/test.csv')
ds = Knee_dataset(df, return_mask=False)
checkpoint = torch.load('session/best_model.pt', map_location='cpu')
model = UNetLext(input_channels=1, output_channels=1)
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

Add saving/visualization logic (see `predict.py`) to produce example predicted
masks for your report/video.


