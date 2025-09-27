import os
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
import numpy as np
from skimage import io, color, transform, filters, feature
from sklearn.svm import SVC
from skimage import exposure
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, top_k_accuracy_score
from joblib import dump, load

# ------------------- CONFIG -------------------
RESOLUTIONS = [(64, 64)]         # matches Combination 11
HOG_PIXELS_PER_CELL = (16, 16)   # larger cell size as in Combination 11
HOG_CELLS_PER_BLOCK = (2, 2)     # same as original
HOG_ORIENTATIONS = 6              # from Combination 11
COLOR_BINS = 16                   # higher color resolution
TEXTURE_BINS = 8                  # as in Combination 11

DATA_DIR = "assignment2_places"
MODEL_FILE = "scene_svm.joblib"
LABEL_FILE = "labels.pkl"

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------- FEATURE FUNCTIONS -------------------
def extract_color_histogram(img, bins=16):
    hist_r, _ = np.histogram(img[:, :, 0], bins=bins, range=(0, 1), density=True)
    hist_g, _ = np.histogram(img[:, :, 1], bins=bins, range=(0, 1), density=True)
    hist_b, _ = np.histogram(img[:, :, 2], bins=bins, range=(0, 1), density=True)
    return np.concatenate([hist_r, hist_g, hist_b])

def extract_edge_features(img_gray):
    edge_sobel = filters.sobel(img_gray)
    hist, _ = np.histogram(edge_sobel, bins=TEXTURE_BINS, range=(0, 1), density=True)
    return hist

def extract_texture_features(img_gray):
    img_gray_uint8 = (img_gray * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(img_gray_uint8, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=TEXTURE_BINS, range=(0, lbp.max() + 1), density=True)
    return hist

def extract_hog_features(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    return feature.hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm="L2-Hys", visualize=False, feature_vector=True)

def extract_multi_features(img, resolutions=RESOLUTIONS):
    img_features = []
    for res in resolutions:
        img_resized = transform.resize(img, res, anti_aliasing=True)
        if img_resized.max() > 2.0:
            img_resized = img_resized / 255.0
        img_gray = color.rgb2gray(img_resized)

        # ---------------- Histogram Equalization ----------------
        img_gray = exposure.equalize_hist(img_gray)
        # --------------------------------------------------------

        feats_color = extract_color_histogram(img_resized, bins=COLOR_BINS)
        feats_edge = extract_edge_features(img_gray)
        feats_texture = extract_texture_features(img_gray)
        feats_hog = extract_hog_features(
            img_gray,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            orientations=HOG_ORIENTATIONS
        )
        combined = np.concatenate([feats_color, feats_edge, feats_texture, feats_hog])
        img_features.append(combined)
    return np.concatenate(img_features)


# ------------------- LOAD DATASET -------------------
def load_dataset(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    X, y = [], []
    for cls_idx, cls in enumerate(classes):
        cls_dir = data_dir / cls
        image_files = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
        for p in tqdm(image_files, desc=f"Processing {cls}"):
            try:
                img = io.imread(p)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                feats = extract_multi_features(img)
                X.append(feats)
                y.append(cls_idx)
            except Exception as e:
                print("Skipping", p, ":", e)
    X = np.vstack(X)
    y = np.array(y)
    return X, y, classes

# ------------------- TRAIN CLASSIFIER -------------------
def train_scene_classifier():
    X, y, classes = load_dataset()
    print("Dataset loaded:", X.shape, y.shape, "Classes:", classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])

    param_grid = {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.001]}
    grid = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=1), n_jobs=1, verbose=2, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print("Best CV score:", grid.best_score_)
    print("Best params:", grid.best_params_)
    print("Hold-out test accuracy:", test_acc)

    # Save model + labels
    dump(best_model, MODEL_FILE)
    with open(LABEL_FILE, "wb") as f:
        pickle.dump(classes, f)
    print(f"Saved model to {MODEL_FILE} and labels to {LABEL_FILE}")

    return best_model, X_test, y_test, classes

# ------------------- MAIN -------------------
if os.path.exists(MODEL_FILE) and os.path.exists(LABEL_FILE):
    print("Loading existing model...")
    model = load(MODEL_FILE)
    with open(LABEL_FILE, "rb") as f:
        classes = pickle.load(f)

    # --- Load dataset and split into train/test to get the unseen test set ---
    X, y, _ = load_dataset()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
else:
    print("Training model...")
    model, X_test, y_test, classes = train_scene_classifier()

# ------------------- EVALUATION -------------------
print("Beginning evaluation on unseen 10% test set...")

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)

# Overall accuracy
acc = np.mean(y_test == y_pred)

# Confusion matrix
Cmat = confusion_matrix(y_test, y_pred)

# Precision, recall, F1
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

# Top-3 overall
topk_overall = top_k_accuracy_score(y_test, y_score, k=3, labels=list(range(len(classes))))

# Top-3 per class
topk_per_class = []
for i in range(len(classes)):
    mask = y_test == i
    topk_per_class.append(top_k_accuracy_score(y_test[mask], y_score[mask], k=3, labels=list(range(len(classes)))))

# ------------------- PRINT RESULTS -------------------
print(f"\nAccuracy: {acc:.4f}")

print("\nConfusion matrix:")
print("Rows = true labels, Columns = predicted labels")
print("Classes:", classes)
print(Cmat)

print("\nPer-class metrics:")
for idx, cls in enumerate(classes):
    print(f"{cls}: Precision={prec[idx]:.4f}, Recall={rec[idx]:.4f}, F1={f1[idx]:.4f}, Top-3={topk_per_class[idx]:.4f}")

print(f"\nTop-3 overall accuracy: {topk_overall:.4f}")

