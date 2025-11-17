import os
import torch
import pandas as pd
import cv2
from torch.utils.data import DataLoader

from dataset import Knee_dataset, read_xray
from model import UNetLext

def main():
    # Load unlabeled test set (masks column may be empty)
    df = pd.read_csv('data/csvs/test.csv')
    dataset = Knee_dataset(df, return_mask=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load best model checkpoint
    checkpoint = torch.load(os.path.join('session', 'best_model.pt'), map_location='cpu')
    state_dict = checkpoint.get('model_state', checkpoint)

    model = UNetLext(input_channels=1, output_channels=1,
                     pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    os.makedirs('predictions', exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            image = batch['image']          # (1,1,H,W)
            img_path = batch['image_path'][0]

            logits = model(image)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            pred_np = preds.squeeze().numpy() * 255.0
            pred_np = pred_np.astype('uint8')

            base_name = os.path.basename(img_path)
            out_path = os.path.join('predictions', f'pred_{base_name}')
            cv2.imwrite(out_path, pred_np)

            if i >= 4:  # just save 5 examples
                break

    print("Saved example predictions to 'predictions/'")

if __name__ == '__main__':
    main()

