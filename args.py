import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Model training/evaluation options')

    parser.add_argument('--csv_dir', type=str, default='data/csvs')
    parser.add_argument('--batch_size', type=int, default=16, choices=[16, 32, 64])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='session')
    parser.add_argument('--csv_path', type=str, default='data/csvs/val.csv',
                        help='CSV path for evaluation scripts.')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('session', 'best_model.pt'),
                        help='Checkpoint path for evaluation/prediction scripts.')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--bar_path', type=str, default='data/csvs/bar_chart.png',
                        help='Output path for evaluation bar charts.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Preferred compute device')

    args = parser.parse_args()
    return args
