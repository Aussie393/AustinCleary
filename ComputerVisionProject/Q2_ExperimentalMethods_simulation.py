import os
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
import itertools
import csv

import numpy as np
from skimage import io, color, transform, filters, feature, exposure
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------- CONFIG -------------------
DATA_DIR = "assignment2_places"
TEST_SIZE = 0.1
RANDOM_STATE = 42
SAVE_DIR = Path(__file__).parent

RESOLUTIONS = [(64, 64), (128, 128)]
HOG_PIXELS_PER_CELL = [(8, 8), (16, 16)]
HOG_CELLS_PER_BLOCK = [(2, 2)]
HOG_ORIENTATIONS = [6, 9]
COLOR_BINS_LIST = [8, 16]
TEXTURE_BINS_LIST = [8, 16]

# ------------------- FEATURE FUNCTIONS -------------------
def extract_color_histogram(img, bins=16):
    hist_r, _ = np.histogram(img[:, :, 0], bins=bins, range=(0, 1), density=True)
    hist_g, _ = np.histogram(img[:, :, 1], bins=bins, range=(0, 1), density=True)
    hist_b, _ = np.histogram(img[:, :, 2], bins=bins, range=(0, 1), density=True)
    return np.concatenate([hist_r, hist_g, hist_b])

def extract_edge_features(img_gray, bins=16):
    edge_sobel = filters.sobel(img_gray)
    hist, _ = np.histogram(edge_sobel, bins=bins, range=(0, 1), density=True)
    return hist

def extract_texture_features(img_gray, bins=16):
    img_gray_uint8 = (img_gray * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(img_gray_uint8, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=bins, range=(0, lbp.max() + 1), density=True)
    return hist

def extract_hog_features(img_gray, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9):
    return feature.hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm="L2-Hys", visualize=False, feature_vector=True)

def extract_multi_features(img, resolutions=[(64,64)], color_bins=16, texture_bins=16,
                           hog_pixels=(8,8), hog_blocks=(2,2), hog_orient=9):
    feats_all = []
    for res in resolutions:
        img_resized = transform.resize(img, res, anti_aliasing=True)
        if img_resized.max() > 2.0:
            img_resized /= 255.0
        img_gray = color.rgb2gray(img_resized)
        img_gray = exposure.equalize_hist(img_gray)
        feats = np.concatenate([
            extract_color_histogram(img_resized, bins=color_bins),
            extract_edge_features(img_gray, bins=texture_bins),
            extract_texture_features(img_gray, bins=texture_bins),
            extract_hog_features(img_gray, pixels_per_cell=hog_pixels,
                                 cells_per_block=hog_blocks, orientations=hog_orient)
        ])
        feats_all.append(feats)
    return np.concatenate(feats_all)

# ------------------- LOAD DATASET -------------------
def load_dataset(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    X, y = [], []
    for cls_idx, cls in enumerate(classes):
        cls_dir = data_dir / cls
        for p in tqdm(sorted(cls_dir.iterdir()), desc=f"Processing {cls}"):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            try:
                img = io.imread(p)
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                X.append(img)
                y.append(cls_idx)
            except:
                continue
    return X, np.array(y), classes

# ------------------- SIMULATION -------------------
def run_simulation():
    X_imgs, y, classes = load_dataset()
    print(f"\nLoaded {len(X_imgs)} images across {len(classes)} classes.")

    X_train_imgs, X_test_imgs, y_train, y_test = train_test_split(
        X_imgs, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results = []
    param_combos = list(itertools.product(
        RESOLUTIONS, COLOR_BINS_LIST, TEXTURE_BINS_LIST,
        HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK, HOG_ORIENTATIONS
    ))

    for idx, (res, c_bins, t_bins, hog_pix, hog_blk, hog_orient) in enumerate(param_combos):
        print(f"\n=== Combination {idx+1}/{len(param_combos)} ===")
        print(f"res={res}, color={c_bins}, texture={t_bins}, hog_pix={hog_pix}, hog_blk={hog_blk}, hog_orient={hog_orient}")

        # Extract features
        X_train = np.array([
            extract_multi_features(img, resolutions=[res], color_bins=c_bins, texture_bins=t_bins,
                                   hog_pixels=hog_pix, hog_blocks=hog_blk, hog_orient=hog_orient)
            for img in tqdm(X_train_imgs, desc="Extracting train features")
        ])
        X_test = np.array([
            extract_multi_features(img, resolutions=[res], color_bins=c_bins, texture_bins=t_bins,
                                   hog_pixels=hog_pix, hog_blocks=hog_blk, hog_orient=hog_orient)
            for img in tqdm(X_test_imgs, desc="Extracting test features")
        ])

        # Train SVM
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1, gamma='scale', probability=True))
        ])
        pipe.fit(X_train, y_train)

        # Evaluate
        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)
        acc = np.mean(y_test == y_pred)
        top3 = top_k_accuracy_score(y_test, y_score, k=3, labels=list(range(len(classes))))
        per_class_top3 = [
            top_k_accuracy_score(y_test[y_test==i], y_score[y_test==i], k=3, labels=list(range(len(classes))))
            for i in range(len(classes))
        ]
        per_class_acc = [
            np.mean(y_pred[y_test==i] == y_test[y_test==i]) if np.any(y_test==i) else 0.0
            for i in range(len(classes))
        ]

        results.append({
            "params": {
                "res": res, "color_bins": c_bins, "texture_bins": t_bins,
                "hog_pix": hog_pix, "hog_blk": hog_blk, "hog_orient": hog_orient
            },
            "acc": acc,
            "top3": top3,
            "per_class_acc": per_class_acc,      # <--- added
            "top3_per_class": per_class_top3
        })

        print(f"Accuracy={acc:.4f}, Top-3={top3:.4f}")

    return results, classes

# ------------------- PLOTTING -------------------
def plot_results(results, classes):
    accs = [r["acc"] for r in results]

    # Overall accuracy across parameter combinations
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(results)), accs, marker='o', label="Accuracy")
    plt.xlabel("Parameter combination index")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs Parameter Combination")
    plt.tight_layout()

    # Per-class accuracy across parameter combinations
    for i, cls in enumerate(classes):
        plt.figure(figsize=(12, 4))
        per_cls_acc = [r["per_class_acc"][i] for r in results]
        plt.plot(range(len(results)), per_cls_acc, marker='o')
        plt.xlabel("Parameter combination index")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy for Class '{cls}'")
        plt.tight_layout()


# ------------------- SAVE CSV -------------------
def save_results_csv(results, classes, save_dir=SAVE_DIR, filename="simulation_results.csv"):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)  # only ensures directory exists (current dir is fine)
    csv_path = save_dir / filename

    param_keys = list(results[0]["params"].keys())
    header = param_keys + ["accuracy", "top3"] + [f"top3_{cls}" for cls in classes]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            row = [r["params"][k] for k in param_keys] + [r["acc"], r["top3"]] + r["top3_per_class"]
            writer.writerow(row)
    print(f"Simulation results saved to {csv_path}")


# ------------------- MAIN -------------------
results, classes = run_simulation()
plot_results(results, classes)
save_results_csv(results, classes)
plt.show()