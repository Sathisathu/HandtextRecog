import torch
from PIL import Image
import torchvision.transforms as T
from model.htr_model import HTRModel
from data_loader.dataset import IAMDataset

# ---------------- Device ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------- Load dataset (for char mappings) ----------------
dataset = IAMDataset(
    images_dir="data/lines",
    labels_file="data/labels.txt",
    img_height=64,
    max_width=256
)
idx_to_char = dataset.idx_to_char
blank_idx = 0

# ---------------- Load model ----------------
num_classes = len(dataset.chars) + 1  # +1 for CTC blank
model = HTRModel(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/htr_model_best.pth", map_location=DEVICE))
model.eval()

# ---------------- Image preprocessing ----------------
def preprocess_image(img_path, img_height=64, max_width=256):
    """
    Preprocess input image for the HTR model:
    - Convert to grayscale
    - Resize keeping aspect ratio
    - Pad to max_width
    - Normalize to [-1, 1]
    """
    img = Image.open(img_path).convert("L")  # grayscale
    w, h = img.size
    new_w = int(w * (img_height / h))  # keep aspect ratio
    new_w = min(new_w, max_width)
    img = img.resize((new_w, img_height))

    # Pad image to max_width
    new_img = Image.new("L", (max_width, img_height), color=255)
    new_img.paste(img, (0, 0))

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    img = transform(new_img).unsqueeze(0)  # [1, 1, H, W]
    return img

# ---------------- CTC greedy decode ----------------
def ctc_decode(output_probs):
    """
    Greedy CTC decode: collapse repeats + remove blanks
    """
    output = output_probs.argmax(2).squeeze(1).detach().cpu().numpy()
    prev = -1
    text = ""
    for idx in output:
        if idx != prev and idx != blank_idx:
            text += idx_to_char.get(idx, "")
        prev = idx
    return text

# ---------------- Prediction ----------------
if __name__ == "__main__":
    # Replace with your image path
    img_path = "my_handwriting/imgFromTrainData.png"  # img path

    img = preprocess_image(img_path).to(DEVICE)

    with torch.no_grad():
        output = model(img)              # [B, W, C]
        output = output.permute(1, 0, 2) # [W, B, C]
        predicted_text = ctc_decode(output)

    print("Predicted text:", predicted_text)
