import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from args import get_args
from dataset import Knee_dataset
from model import UNetLext
from utils import dice_loss_from_logits


def _dice_per_sample(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def evaluate_model(model, loader, device):
    bce = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    total_batches = 0
    per_image_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            logits = model(images)

            loss_bce = bce(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = loss_bce + loss_dice

            total_loss += loss.item()
            total_batches += 1

            sample_dice = _dice_per_sample(logits, masks).cpu().tolist()
            for path, dice_score in zip(batch['image_path'], sample_dice):
                per_image_scores.append((os.path.basename(path), dice_score))

    avg_loss = total_loss / max(total_batches, 1)
    avg_dice = sum(score for _, score in per_image_scores) / max(len(per_image_scores), 1)
    return avg_loss, avg_dice, per_image_scores


def plot_scores(per_image_scores, out_path):
    if not per_image_scores:
        print('No scores to plot.')
        return

    labels = [name for name, _ in per_image_scores]
    scores = [score for _, score in per_image_scores]
    indices = list(range(len(scores)))

    plt.figure(figsize=(12, 5))
    plt.bar(indices, scores, color='#4C72B0')
    plt.xticks(indices, labels, rotation=90, fontsize=6)
    plt.ylabel('Dice Score')
    plt.xlabel('Validation Images')
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    dir_name = os.path.dirname(out_path) or '.'
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Saved bar chart to {out_path}')


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found at {args.checkpoint}')

    df = pd.read_csv(args.csv_path)
    dataset = Knee_dataset(df, return_mask=True)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)

    model = UNetLext(input_channels=args.input_channels,
                     output_channels=args.output_channels,
                     pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)

    avg_loss, avg_dice, per_image_scores = evaluate_model(model, loader, device)
    print(f'Average loss: {avg_loss:.4f}')
    print(f'Average Dice: {avg_dice:.4f}')

    plot_scores(per_image_scores, args.bar_path)


if __name__ == '__main__':
    main()
