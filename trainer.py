from args import get_args
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import dice_loss_from_logits, dice_score_from_logits


def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)

    os.makedirs(args.out_dir, exist_ok=True)
    best_dice = -1.0
    best_path = os.path.join(args.out_dir, 'best_model.pt')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for data_batch in train_loader:
            images = data_batch['image'].to(device = device)
            masks = data_batch['mask'].to(device = device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_bce = bce(outputs, masks)

            loss_dice = dice_loss_from_logits(outputs, masks)

            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss/ len(train_loader)    

        # vadidation
        val_loss, val_score = validate_model(model, val_loader, bce, device)
        print(f"echos{epoch + 1}/{args.epochs}")
        print(f"Train loss {train_loss:.4f}")
        print(f"val loss {val_loss:.4f}")
        print(f"Val dice {val_score:.4f}")

        if val_score > best_dice:
            best_dice = val_score
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_dice': val_score,
                    'train_loss': train_loss
                },
                best_path
            )
            print(f"Saved new best model to {best_path}")

    last_path = os.path.join(args.out_dir, 'last_model.pt')
    torch.save(
        {
            'model_state': model.state_dict(),
            'epoch': args.epochs,
            'val_dice': val_score,
            'train_loss': train_loss
        },
        last_path
    )
    print(f"Saved final model to {last_path}")

def  validate_model(model, val_loader,bce, device):
    model.eval()
    val_loss = 0.0
    val_score = 0.0

    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch['image'].to(device=device)
            masks = data_batch['mask'].to(device=device)
            outputs = model(images)
            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            val_loss += loss.item()
            val_score += dice_score_from_logits(outputs, masks)

    val_epoch_loss = val_loss / len(val_loader)
    val_epoch_dice = val_score / len(val_loader)
    return val_epoch_loss, val_epoch_dice 
        
