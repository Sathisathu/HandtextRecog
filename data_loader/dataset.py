import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class IAMDataset(Dataset):
    def __init__(self, images_dir="data/lines", labels_file="data/labels.txt", img_height=64, max_width=None):
        """
        Args:
            images_dir: Root folder containing IAM line images
            labels_file: A txt file with format -> relative_image_path \t transcription
            img_height: Fixed height to resize images
            max_width: Optional max width to limit very wide images
        """
        self.images_dir = images_dir
        self.img_height = img_height
        self.max_width = max_width

        # Load (image_path, text) pairs
        self.samples = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                if "\t" not in line:
                    continue
                fname, text = line.strip().split("\t", 1)
                self.samples.append((fname, text))

        # Build vocabulary (all unique characters)
        self.chars = sorted(set("".join([s[1] for s in self.samples])))
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}  # +1 for CTC blank
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.blank_idx = 0  # reserved for CTC blank

        # Preprocessing transform (grayscale + tensor + normalize)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, text = self.samples[idx]

        # First try direct path
        img_path = os.path.join(self.images_dir, fname)

        # If not found, search recursively
        if not os.path.exists(img_path):
            found = False
            for root, _, files in os.walk(self.images_dir):
                if fname in files:
                    img_path = os.path.join(root, fname)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Cannot find image file: {fname} in {self.images_dir}")

        # Load image
        img = Image.open(img_path).convert("L")
        w, h = img.size
        new_w = int(w * (self.img_height / h))
        if self.max_width:
            new_w = min(new_w, self.max_width)
        img = img.resize((new_w, self.img_height))

        img = self.transform(img)

        # Encode label (string -> int sequence)
        label = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
        label = torch.tensor(label, dtype=torch.long)

        return img, label, len(label)


# 🔹 Collate function
def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)

    # Pad images by max width in batch
    max_w = max(img.shape[2] for img in imgs)
    imgs_padded = []
    for img in imgs:
        pad_w = max_w - img.shape[2]
        pad = torch.nn.functional.pad(img, (0, pad_w, 0, 0), value=0)
        imgs_padded.append(pad)
    imgs_padded = torch.stack(imgs_padded)

    # Concatenate labels
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor(lengths, dtype=torch.long)

    return imgs_padded, labels_concat, label_lengths
