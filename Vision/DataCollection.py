import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import albumentations as alb

# ----------------------------- #
#  GPU SAFE MEMORY GROWTH
# ----------------------------- #
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ----------------------------- #
#  HELPER: SAFE LOAD IMAGE
# ----------------------------- #
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"[ERROR] Failed to load image: {path}")
    return img

# ----------------------------- #
#  MOVE LABELS INTO train/test/val
# ----------------------------- #
def move_labels():
    print("\n📦 Moving labels into train/test/val/...")

    for folder in ["train", "test", "val"]:
        images_path = os.path.join("Data", folder, "images")
        for file in os.listdir(images_path):
            json_name = file.split('.')[0] + ".json"

            src = os.path.join("Data", "labels", json_name)
            dst = os.path.join("Data", folder, "labels", json_name)

            if os.path.exists(src):
                os.replace(src, dst)
                print(f"  ✔ moved {json_name} → {folder}/labels/")
            # else: silently ignore missing

# ----------------------------- #
#  AUGMENTOR CONFIG
# ----------------------------- #
augmentor = alb.Compose(
    [
        alb.RandomCrop(width=450, height=450),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
        alb.RandomGamma(p=0.2),
        alb.RGBShift(p=0.2),
        alb.VerticalFlip(p=0.5),
    ],
    keypoint_params=alb.KeypointParams(format="xy", label_fields=["class_labels"]),
)

# ----------------------------- #
#  READ LABELME POINTS
# ----------------------------- #
def read_keypoints(json_path):
    if not os.path.exists(json_path):
        return [0, 0], [0, 0], [0, 0, 0, 0], [0, 0]

    data = json.load(open(json_path))

    left = [0, 0]
    right = [0, 0]
    classes = [0, 0]
    coords = [0, 0, 0, 0]

    for shape in data["shapes"]:
        label = shape["label"]
        x, y = shape["points"][0]

        if label == "LeftEye":
            classes[0] = 1
            left = [x, y]

        if label == "RightEye":
            classes[1] = 1
            right = [x, y]

    coords = left + right
    return left, right, coords, classes

# ----------------------------- #
#  AUGMENT DATASET
# ----------------------------- #
def augment_dataset():
    print("\n🧪 Starting augmentation...")

    for split in ["train", "test", "val"]:
        img_dir = os.path.join("Data", split, "images")
        print(f"\n🔹 Augmenting: {split}")

        for img_name in tqdm(os.listdir(img_dir), desc=f"{split} images"):

            img_path = os.path.join(img_dir, img_name)
            json_path = os.path.join("Data", split, "labels", img_name.replace(".jpg", ".json"))

            try:
                img = load_image(img_path)
            except Exception as e:
                print(f"[ERROR] Skipping corrupted image {img_name}: {e}")
                continue

            left, right, coords, classes = read_keypoints(json_path)

            # Normalize original keypoints (before augmentation)
            H, W = img.shape[:2]
            norm_coords = list(np.divide(coords, [W, H, W, H]))

            for i in range(40):  # generate 120 augmentations
                try:
                    keypoints = [left, right]
                    augmented = augmentor(image=img, keypoints=keypoints, class_labels=["LeftEye", "RightEye"])

                    # Save image
                    save_img = os.path.join("aug_Data", split, "images", f"{img_name.split('.')[0]}.{i}.jpg")
                    cv2.imwrite(save_img, augmented["image"])

                    # Build annotation JSON
                    annot = {
                        "image": img_name,
                        "class": [0, 0],
                        "keypoints": [0, 0, 0, 0],
                    }

                    for idx, cl in enumerate(augmented["class_labels"]):
                        if cl == "LeftEye":
                            annot["class"][0] = 1
                            annot["keypoints"][0] = augmented["keypoints"][idx][0]
                            annot["keypoints"][1] = augmented["keypoints"][idx][1]

                        if cl == "RightEye":
                            annot["class"][1] = 1
                            annot["keypoints"][2] = augmented["keypoints"][idx][0]
                            annot["keypoints"][3] = augmented["keypoints"][idx][1]

                    # Normalize new keypoints by 450×450
                    annot["keypoints"] = list(np.divide(annot["keypoints"], [450, 450, 450, 450])) #kemungkinan diubah nanti

                    # Save annotation JSON
                    save_json = os.path.join("aug_Data", split, "labels", f"{img_name.split('.')[0]}.{i}.json")
                    json.dump(annot, open(save_json, "w"))

                except Exception as e:
                    print(f"[ERROR] Failed augmentation for {img_name}: {e}")
                    continue

    print("\n🎉 Augmentation complete!")

# ----------------------------- #
# RUN EVERYTHING
# ----------------------------- #
if __name__ == "__main__":
    move_labels()
    augment_dataset()
