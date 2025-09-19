if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from data_loader.dataset import IAMDataset, collate_fn
    from model.htr_model import HTRModel
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    from torch.cuda.amp import autocast, GradScaler
    import os

    # -------- Hyperparameters --------
    BATCH_SIZE = 16
    IMG_HEIGHT = 64
    MAX_IMG_WIDTH = 256
    EPOCHS = 25
    LR = 1e-3
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------- Device --------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # -------- Dataset & DataLoader --------
    dataset = IAMDataset(
        images_dir="data/lines",
        labels_file="data/labels.txt",
        img_height=IMG_HEIGHT,
        max_width=MAX_IMG_WIDTH
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, num_workers=4)

    # -------- Model --------
    num_classes = len(dataset.chars) + 1
    model = HTRModel(num_classes=num_classes).to(DEVICE)

    # -------- Loss & Optimizer --------
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    # -------- Training Loop --------
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

        for imgs, labels, label_lengths in progress_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                outputs = model(imgs)              # [B, W, C]
                outputs = outputs.permute(1, 0, 2) # [W, B, C]
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(DEVICE)

            loss = ctc_loss(outputs.float(), labels, input_lengths, label_lengths.to(DEVICE))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(SAVE_DIR, f"htr_model_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model: {save_path}")

        # Optional: save every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(SAVE_DIR, f"htr_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

    print("Deep training complete!")
