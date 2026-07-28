import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from tqdm import tqdm
import os
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import get_loaders, save_checkpoint, load_checkpoint, check_accuracy
import kagglehub

path = kagglehub.dataset_download("yaroslavchyrko/rescuenet")

#hyperparameters

alpha = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 3
num_workers = 2
image_height =160
image_width = 240
pin_memory = True
load_model = True
train_img_dir = os.path.join(path, "train", "train-org-img")
train_mask_dir = os.path.join(path, "train", "train-label-img")
val_img_dir = os.path.join(path, "val", "val-org-img")
val_mask_dir = os.path.join(path, "val", "val-label-img")

def train(loader, model, optimizer, loss_func, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        with torch.amp.autocast(device_type=device):
            predictions = model(data)
            loss = loss_func(predictions, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

def main():
    train_transforms = v2.Compose([
        v2.Resize((image_height, image_width)),
        v2.RandomRotation(degrees=35),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.1),
        v2.ToImage(),
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
    ])

    val_transforms = v2.Compose([
        v2.Resize((image_height, image_width)),
        v2.ToImage(),
        v2.ToDtype({tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
    ])

    model = UNET(in_channels=3, out_channels=3).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    train_loader, val_loader = get_loaders(train_img_dir, train_mask_dir, 
                                           val_img_dir, val_mask_dir, 
                                           batch_size, 
                                           train_transforms, val_transforms, 
                                           num_workers, pin_memory)

    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, loss_func, scaler)

        checkpoint = {"state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

        check_accuracy(val_loader,model,device=device)

if __name__ == "__main__":
    main()