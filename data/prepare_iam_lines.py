import os
import shutil

def prepare_iam_lines(iam_root, out_images="images/", out_labels="labels.txt"):
    os.makedirs(out_images, exist_ok=True)

    lines_txt = os.path.join(iam_root, "ascii", "lines.txt")
    if not os.path.exists(lines_txt):
        raise FileNotFoundError(f"Cannot find {lines_txt}. Check IAM root path: {iam_root}")

    labels = []
    # Build a dict from line‐id to transcription
    id_to_transcription = {}
    with open(lines_txt, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            line_id, status = parts[0], parts[1]
            if status != "ok":
                continue
            transcription = " ".join(parts[8:])
            if transcription.strip() == "":
                continue
            id_to_transcription[line_id] = transcription

    # Now walk through the nested image folders in iam_root/lines
    lines_img_root = os.path.join(iam_root, "lines")
    count = 0
    for subdir, dirs, files in os.walk(lines_img_root):
        for fname in files:
            if not fname.lower().endswith(".png"):
                continue
            # build line_id from filename (without .png)
            line_id = os.path.splitext(fname)[0]
            if line_id in id_to_transcription:
                src_path = os.path.join(subdir, fname)
                dst_path = os.path.join(out_images, fname)
                # Copy image
                shutil.copy(src_path, dst_path)
                # Write label
                transcription = id_to_transcription[line_id]
                labels.append(f"{fname}\t{transcription}\n")
                count += 1

    with open(out_labels, "w", encoding="utf-8") as f:
        f.writelines(labels)

    print(f"Prepared {count} line images.")
    print(f"Images in: {out_images}")
    print(f"Labels file: {out_labels}")

if __name__ == "__main__":
    iam_root = "."   # because script is already inside "data"
    prepare_iam_lines(iam_root)

