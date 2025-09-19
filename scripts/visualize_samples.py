import matplotlib.pyplot as plt
import os

def show_samples(images_dir="data/images", labels_file="data/labels.txt", num_samples=5):
    # Load labels
    with open(labels_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    samples = lines[:num_samples]

    plt.figure(figsize=(15, 5))
    for i, line in enumerate(samples):
        fname, text = line.strip().split("\t", 1)
        img_path = os.path.join(images_dir, fname)
        img = plt.imread(img_path)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(text, fontsize=10)
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    show_samples()
