from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import cKDTree

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "assignment2_stereodata"
IMGS_L = DATA_DIR / "images_left"
IMGS_R = DATA_DIR / "images_right"
STEREO_CALIB = DATA_DIR / "stereo_calib_diver.pkl"
CAM_POSES = DATA_DIR / "camera_pose_data.pkl"
TERRAIN = DATA_DIR / "terrain_data.pkl"

# ----------------------------
# Utility functions
# ----------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_calib(calib_dict):
    if not isinstance(calib_dict, dict):
        try:
            K_left = calib_dict.K_left
            K_right = calib_dict.K_right
            R_lr = calib_dict.R_lr
            T_lr = calib_dict.T_lr
            dist_left = calib_dict.dist_left
            dist_right = calib_dict.dist_right
            return (np.array(K_left), np.array(dist_left).reshape(-1), np.array(K_right),
                    np.array(dist_right).reshape(-1), np.array(R_lr), np.array(T_lr).reshape(3,1))
        except Exception:
            raise ValueError("Unsupported calibration structure and not a dict.")

    keys = {k.lower(): k for k in calib_dict.keys()}
    def get_key(poss):
        for p in poss:
            if p.lower() in keys:
                return calib_dict[keys[p.lower()]]
        return None

    K_left = get_key(['K_left', 'K1', 'K_l', 'Kleft'])
    K_right = get_key(['K_right', 'K2', 'K_r', 'Kright'])
    dist_left = get_key(['dist_left', 'dist1', 'distcoeffs1', 'dist_l'])
    dist_right = get_key(['dist_right', 'dist2', 'distcoeffs2', 'dist_r'])
    R_lr = get_key(['R_lr', 'R', 'R_left_right', 'Rlr'])
    T_lr = get_key(['T_lr', 'T', 't_lr', 't'])

    if K_left is None or K_right is None:
        for v in calib_dict.values():
            arr = np.array(v)
            if arr.shape == (3,3) and K_left is None:
                K_left = arr
            elif arr.shape == (3,3) and K_right is None:
                K_right = arr
    if K_left is None or K_right is None:
        raise KeyError('Could not locate K_left/K_right in calibration pickle.')

    if dist_left is None:
        dist_left = np.zeros((5,))
    if dist_right is None:
        dist_right = np.zeros((5,))
    if R_lr is None or T_lr is None:
        raise KeyError('Could not locate stereo relative pose R/T in calibration pickle.')

    return (np.array(K_left, dtype=np.float64), np.array(dist_left, dtype=np.float64).reshape(-1),
            np.array(K_right, dtype=np.float64), np.array(dist_right, dtype=np.float64).reshape(-1),
            np.array(R_lr, dtype=np.float64), np.array(T_lr, dtype=np.float64).reshape(3,1))

def build_proj_matrices(Kl, Kr, Rlr, Tlr):
    P_left = Kl @ np.hstack((np.eye(3), np.zeros((3,1))))
    P_right = Kr @ np.hstack((Rlr, Tlr))
    return P_left, P_right

# ----------------------------
# Load data
# ----------------------------
calib = load_pickle(STEREO_CALIB)
K_left, dist_left, K_right, dist_right, R_lr, T_lr = parse_calib(calib)
P_left, P_right = build_proj_matrices(K_left, K_right, R_lr, T_lr)
print("Loaded stereo calibration.")

cam_data = load_pickle(CAM_POSES)
print("Camera pose keys:", list(cam_data.keys()))

# Handle shapes
R_all = np.array(cam_data['R'])
t_all = np.array(cam_data['t'])
if R_all.shape[0] == 3 and R_all.shape[1] == 3 and R_all.ndim == 3:
    R_all = np.transpose(R_all, (2,0,1))
elif R_all.ndim == 3 and R_all.shape[1:] == (3,3):
    pass
else:
    raise ValueError('Unexpected R shape: ' + str(R_all.shape))

if t_all.ndim == 2 and t_all.shape[0] == 3:
    t_all = t_all.T
elif t_all.ndim == 3 and t_all.shape[0] == 3:
    t_all = np.transpose(t_all, (2,0,1)).reshape(-1,3)

n_frames = R_all.shape[0]
print(f"Found {n_frames} camera poses.")

files_left = cam_data.get('filenames_left', sorted([p.name for p in IMGS_L.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg','.pgm']]))
files_right = cam_data.get('filenames_right', sorted([p.name for p in IMGS_R.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg','.pgm']]))

n_iter = min(len(files_left), len(files_right), n_frames)
print(f"Processing {n_iter} frames (min of files and poses).")

# ----------------------------
# Process first stereo pair
# ----------------------------
imgL_rgb = cv2.cvtColor(cv2.imread(str(IMGS_L / files_left[0])), cv2.COLOR_BGR2RGB)
imgR_rgb = cv2.cvtColor(cv2.imread(str(IMGS_R / files_right[0])), cv2.COLOR_BGR2RGB)
imgL_gray = cv2.cvtColor(imgL_rgb, cv2.COLOR_RGB2GRAY)
imgR_gray = cv2.cvtColor(imgR_rgb, cv2.COLOR_RGB2GRAY)

# ----------------------------
# Rectification maps
# ----------------------------
def compute_rectification_maps(imgL, imgR, Kl, Dl, Kr, Dr, R_lr, T_lr, alpha=0):
    H, W = imgL.shape[:2]
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(Kl, Dl, Kr, Dr, (W,H), R_lr, T_lr,
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    mapL1, mapL2 = cv2.initUndistortRectifyMap(Kl, Dl, RL, PL, (W,H), cv2.CV_16SC2)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(Kr, Dr, RR, PR, (W,H), cv2.CV_16SC2)
    return mapL1, mapL2, mapR1, mapR2, RL, RR, PL, PR, Q

def rectify_images(imgL, imgR, mapL1, mapL2, mapR1, mapR2):
    imgLr = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR)
    imgRr = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)
    return imgLr, imgRr

# ----------------------------
# Feature matching (ORB/SIFT)
# ----------------------------
def match_features_ORB(imgL, imgR, nfeatures=4000, ratio=0.85):
    gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kpl, desl = orb.detectAndCompute(gL, None)
    kpr, desr = orb.detectAndCompute(gR, None)
    if desl is None or desr is None or len(kpl)==0 or len(kpr)==0:
        return np.empty((0,2)), np.empty((0,2))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(desl, desr, k=2)
    cand = [m for m,n in knn if n is not None and m.distance < ratio*n.distance]
    if len(cand)<8:
        return np.empty((0,2)), np.empty((0,2))
    ptsL = np.float32([kpl[m.queryIdx].pt for m in cand])
    ptsR = np.float32([kpr[m.trainIdx].pt for m in cand])
    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 1.5, 0.999)
    if mask is None:
        return np.empty((0,2)), np.empty((0,2))
    inliers = mask.ravel().astype(bool)
    return ptsL[inliers], ptsR[inliers]

def match_features_SIFT(imgL, imgR, nfeatures=4000, ratio=0.75):
    gL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kpl, desl = sift.detectAndCompute(gL, None)
    kpr, desr = sift.detectAndCompute(gR, None)
    if desl is None or desr is None or len(kpl)==0 or len(kpr)==0:
        return np.empty((0,2)), np.empty((0,2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desl, desr, k=2)
    cand = [m for m,n in knn if n is not None and m.distance < ratio*n.distance]
    if len(cand)<8:
        return np.empty((0,2)), np.empty((0,2))
    ptsL = np.float32([kpl[m.queryIdx].pt for m in cand])
    ptsR = np.float32([kpr[m.trainIdx].pt for m in cand])
    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 1.5, 0.999)
    if mask is None:
        return np.empty((0,2)), np.empty((0,2))
    inliers = mask.ravel().astype(bool)
    return ptsL[inliers], ptsR[inliers]

# ----------------------------
# Triangulate points in left camera frame
# ----------------------------
def triangulate_left_camera(ptsL, ptsR, PL, PR, RL=None):
    if len(ptsL) == 0:
        return np.empty((0,3))
    Xh = cv2.triangulatePoints(PL, PR, ptsL.T, ptsR.T)
    X = (Xh[:3]/Xh[3]).T
    if RL is not None:
        X = (RL.T @ X.T).T
    return X

# ----------------------------
# Transform to world frame
# ----------------------------
def left_to_world(X_L, R_wc, t_wc):
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc.reshape(3,1)
    return (R_cw @ X_L.T + t_cw).T

def detect_match_and_plot(imgL_gray, imgR_gray, imgL_rgb, imgR_rgb, method="SIFT"):
    # Create detector + matcher
    if method == "SIFT":
        detector = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=5000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError("Method must be 'SIFT' or 'ORB'")

    # Detect keypoints
    kpL, desL = detector.detectAndCompute(imgL_gray, None)
    kpR, desR = detector.detectAndCompute(imgR_gray, None)
    print(f"{method} detected {len(kpL)} keypoints in LEFT and {len(kpR)} in RIGHT")

    # Match
    matches = bf.match(desL, desR)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"{method} found {len(matches)} matches")

    # ----------------------------
    # Figure 1: keypoints in LEFT + RIGHT
    # ----------------------------
    ptsL = np.array([k.pt for k in kpL], dtype=np.float32)
    ptsR = np.array([k.pt for k in kpR], dtype=np.float32)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(imgL_rgb)
    axs[0].scatter(ptsL[:, 0], ptsL[:, 1], c="red", s=8, marker="o", alpha=0.6)
    axs[0].set_title(f"{method} keypoints - LEFT")
    axs[0].axis("off")

    axs[1].imshow(imgR_rgb)
    axs[1].scatter(ptsR[:, 0], ptsR[:, 1], c="red", s=8, marker="o", alpha=0.6)
    axs[1].set_title(f"{method} keypoints - RIGHT")
    axs[1].axis("off")

    # ----------------------------
    # Figure 2: all matches as lines
    # ----------------------------
    img_concat = np.hstack((imgL_rgb, imgR_rgb))
    plt.figure(figsize=(15, 6))
    plt.imshow(img_concat)

    for m in matches:
        ptL = tuple(map(int, kpL[m.queryIdx].pt))
        ptR = tuple(map(int, kpR[m.trainIdx].pt))
        ptR_shifted = (ptR[0] + imgL_rgb.shape[1], ptR[1])

        # Draw keypoints
        plt.scatter(*ptL, c="yellow", s=10)
        plt.scatter(*ptR_shifted, c="yellow", s=10)

        # Draw line connecting matched points
        plt.plot([ptL[0], ptR_shifted[0]], [ptL[1], ptR_shifted[1]],
                 c="lime", linewidth=0.5, alpha=0.7)

    plt.title(f"{method} matches (all matched points)")
    plt.axis("off")
    plt.show()

# ----------------------------
# Robust triangulation
# ----------------------------
def robust_triangulation(imgL_gray, imgR_gray, imgL_rgb, imgR_rgb, K_left, K_right, R_lr, T_lr, method="SIFT", ratio_thresh=0.75):
    if method == "SIFT":
        detector = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2)
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=5000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        raise ValueError("Method must be 'SIFT' or 'ORB'")

    # Detect keypoints
    kpL, desL = detector.detectAndCompute(imgL_gray, None)
    kpR, desR = detector.detectAndCompute(imgR_gray, None)
    print(f"{method}: detected {len(kpL)} keypoints LEFT, {len(kpR)} keypoints RIGHT")

    # KNN match + ratio test
    knn_matches = bf.knnMatch(desL, desR, k=2)
    good_matches = [m for m,n in knn_matches if m.distance < ratio_thresh * n.distance]
    print(f"{method}: {len(good_matches)} matches after ratio test")

    if len(good_matches) < 8:
        print(f"Not enough matches for RANSAC. Skipping {method}")
        return

    ptsL = np.float32([kpL[m.queryIdx].pt for m in good_matches])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in good_matches])

    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 3.0, 0.99)
    inlier_mask = mask.ravel().astype(bool)
    print(f"{method}: {np.sum(inlier_mask)} inlier matches after RANSAC")

    # Plot inlier matches
    img_concat = np.hstack((imgL_rgb, imgR_rgb))
    plt.figure(figsize=(15, 6))
    plt.imshow(img_concat)
    inlier_matches = [m for i,m in enumerate(good_matches) if inlier_mask[i]]

    for m in inlier_matches:
        ptL = tuple(map(int, kpL[m.queryIdx].pt))
        ptR = tuple(map(int, kpR[m.trainIdx].pt))
        ptR_shifted = (ptR[0] + imgL_rgb.shape[1], ptR[1])
        plt.scatter(*ptL, c="yellow", s=10)
        plt.scatter(*ptR_shifted, c="yellow", s=10)
        plt.plot([ptL[0], ptR_shifted[0]], [ptL[1], ptR_shifted[1]], c="lime", linewidth=0.5, alpha=0.7)
    plt.title(f"{method} inlier matches (ratio test + RANSAC)")
    plt.axis("off")
    plt.show()

    # Triangulate inlier points
    P_left = K_left @ np.hstack((np.eye(3), np.zeros((3,1))))
    P_right = K_right @ np.hstack((R_lr, T_lr))

    ptsL_in = np.float32([kpL[m.queryIdx].pt for i,m in enumerate(good_matches) if inlier_mask[i]]).T
    ptsR_in = np.float32([kpR[m.trainIdx].pt for i,m in enumerate(good_matches) if inlier_mask[i]]).T
    pts4 = cv2.triangulatePoints(P_left, P_right, ptsL_in, ptsR_in)
    pts3 = (pts4[:3] / pts4[3]).T

    # Colors from left image
    pts_rgb = np.array([kpL[m.queryIdx].pt for i,m in enumerate(good_matches) if inlier_mask[i]], dtype=np.int32)
    pts_rgb[:,0] = np.clip(pts_rgb[:,0], 0, imgL_rgb.shape[1]-1)
    pts_rgb[:,1] = np.clip(pts_rgb[:,1], 0, imgL_rgb.shape[0]-1)
    colors = imgL_rgb[pts_rgb[:,1], pts_rgb[:,0]]

    # Plot 3D points
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts3[:,0], pts3[:,1], pts3[:,2], c=colors/255.0, s=2)
    ax.set_title(f"{method} 3D points (inliers only)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

# ----------------------------
# Control flag
# ----------------------------
variable_run_intermediate_steps = False  # Set to False to skip intermediate visualizations

if variable_run_intermediate_steps:
    for method in ["ORB", "SIFT"]:
        detect_match_and_plot(imgL_gray, imgR_gray, imgL_rgb, imgR_rgb, method=method)
        robust_triangulation(imgL_gray, imgR_gray, imgL_rgb, imgR_rgb, K_left, K_right, R_lr, T_lr, method=method)

def run_reconstruction(detector_name: str = "SIFT"):
    """
    Reconstruct a 3D point cloud from diver-rig stereo images using ORB or SIFT.

    Parameters
    ----------
    detector_name : str
        Feature detector to use ("ORB" or "SIFT")

    Returns
    -------
    X_world : np.ndarray
        Nx3 array of 3D points in world coordinates
    colors : np.ndarray or None
        Nx3 array of RGB colors (0-255) from left images
    """
    # Load calibration
    calib = load_pickle(STEREO_CALIB)
    K_left, dist_left, K_right, dist_right, R_lr, T_lr = parse_calib(calib)

    # Load camera poses
    cam_data = load_pickle(CAM_POSES)
    R_all = np.array(cam_data['R'])
    t_all = np.array(cam_data['t'])
    if R_all.shape[:2] == (3,3) and R_all.ndim == 3:
        R_all = np.transpose(R_all, (2,0,1))
    if t_all.ndim == 2 and t_all.shape[0] == 3:
        t_all = t_all.T

    # Load images
    files_left = sorted([p.name for p in IMGS_L.iterdir() if p.suffix.lower() in ['.png','.jpg']])
    files_right = sorted([p.name for p in IMGS_R.iterdir() if p.suffix.lower() in ['.png','.jpg']])
    n_iter = min(len(files_left), len(files_right), R_all.shape[0])
    print(f"Processing {n_iter} stereo pairs...")

    imgsL = [cv2.imread(str(IMGS_L/f)) for f in files_left[:n_iter]]
    imgsR = [cv2.imread(str(IMGS_R/f)) for f in files_right[:n_iter]]

    all_Xw, all_colors = [], []

    # Choose matcher
    match_fn = match_features_SIFT if detector_name.upper() == "SIFT" else match_features_ORB

    for i in range(n_iter):
        print(f"Processing frame {i+1}/{n_iter}")
        imgL, imgR = imgsL[i], imgsR[i]

        # Rectify
        mapL1, mapL2, mapR1, mapR2, RL, RR, PL, PR, Q = compute_rectification_maps(
            imgL, imgR, K_left, dist_left, K_right, dist_right, R_lr, T_lr, alpha=0
        )
        imgLr, imgRr = rectify_images(imgL, imgR, mapL1, mapL2, mapR1, mapR2)

        # Feature matching
        ptsL, ptsR = match_fn(imgLr, imgRr)
        print(f"{detector_name} Frame {i}: {len(ptsL)} inlier matches") 

        if len(ptsL) < 8:
            continue  # Skip frame

        # Triangulate in left-camera frame
        X_L = triangulate_left_camera(ptsL, ptsR, PL, PR, RL)

        # Cheirality check
        XR = (R_lr @ X_L.T + T_lr).T
        z_ok = (X_L[:,2] > 0) & (XR[:,2] > 0)
        X_L = X_L[z_ok]
        ptsL_ok = ptsL[z_ok]

        # Transform to world frame
        Xw = left_to_world(X_L, R_all[i], t_all[i])
        all_Xw.append(Xw)

        # Colors
        if len(ptsL_ok) > 0:
            px = np.clip(ptsL_ok[:,0].astype(int), 0, imgLr.shape[1]-1)
            py = np.clip(ptsL_ok[:,1].astype(int), 0, imgLr.shape[0]-1)
            colors = imgLr[py, px, ::-1]  # BGR -> RGB
            all_colors.append(colors)

    X_world = np.vstack(all_Xw) if all_Xw else np.empty((0,3))
    colors = np.vstack(all_colors) if all_colors else None
    print("Triangulation complete. Total points:", X_world.shape[0])
    
    return X_world, colors

def plot_orb_sift_clouds(X_orb, X_sift, colors_orb=None, colors_sift=None):
    """
    Plot ORB and SIFT point clouds individually in 3D.

    Parameters
    ----------
    X_orb : np.ndarray
        Nx3 array of ORB 3D points.
    X_sift : np.ndarray
        Mx3 array of SIFT 3D points.
    colors_orb : np.ndarray or None
        Nx3 array of RGB colors for ORB points (0-255). If None, uses blue.
    colors_sift : np.ndarray or None
        Mx3 array of RGB colors for SIFT points (0-255). If None, uses red.
    """

    # ----------------------------
    # ORB Point Cloud
    # ----------------------------
    if X_orb is not None and X_orb.shape[0] > 0:
        fig_orb = plt.figure(figsize=(8, 6))
        ax_orb = fig_orb.add_subplot(111, projection="3d")
        ax_orb.scatter(
            X_orb[:,0], X_orb[:,1], X_orb[:,2],
            c=colors_orb/255.0 if colors_orb is not None else "b",
            s=1
        )
        ax_orb.set_title("ORB Point Cloud")
        ax_orb.set_xlabel("X")
        ax_orb.set_ylabel("Y")
        ax_orb.set_zlabel("Z")

    # ----------------------------
    # SIFT Point Cloud
    # ----------------------------
    if X_sift is not None and X_sift.shape[0] > 0:
        fig_sift = plt.figure(figsize=(8, 6))
        ax_sift = fig_sift.add_subplot(111, projection="3d")
        ax_sift.scatter(
            X_sift[:,0], X_sift[:,1], X_sift[:,2],
            c=colors_sift/255.0 if colors_sift is not None else "r",
            s=1
        )
        ax_sift.set_title("SIFT Point Cloud")
        ax_sift.set_xlabel("X")
        ax_sift.set_ylabel("Y")
        ax_sift.set_zlabel("Z")
        plt.show()

# ----------------------------
# Run both ORB and SIFT, then plot side by side
# ----------------------------
X_orb, colors_orb = run_reconstruction("ORB")
X_sift, colors_sift = run_reconstruction("SIFT")

plot_orb_sift_clouds(X_orb, X_sift, colors_orb, colors_sift)

# ----------------------------
# Load terrain ground truth
# ----------------------------
# Load reference terrain points
with open(TERRAIN, "rb") as f:
    terrain_data = pickle.load(f)

X, Y = np.meshgrid(terrain_data['X'], terrain_data['Y'])
Z = -terrain_data['height_grid']
terrain_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

print(f"Loaded terrain ground truth with {terrain_points.shape[0]} points.")

# ----------------------------
# Compare function
# ----------------------------
def compare_and_visualize_colored(recon_cloud: np.ndarray, terrain_points: np.ndarray, name: str="Reconstruction"):
    """
    Compare a reconstructed point cloud to a reference terrain and visualize with color-coded errors.
    Includes histogram, 3D plot, and XY/XZ/YZ projections.
    """
    # Clean inputs
    recon_cloud = recon_cloud[np.isfinite(recon_cloud).all(axis=1)]
    terrain_points = terrain_points[np.isfinite(terrain_points).all(axis=1)]

    if recon_cloud.shape[0] == 0:
        print(f"{name}: no valid points to compare.")
        return None

    # Compute distances to terrain
    tree = cKDTree(terrain_points)
    distances, _ = tree.query(recon_cloud)

    # --- Summary ---
    print(f"{name} -> Reference terrain distances (meters):")
    print(f"  Mean:   {distances.mean():.3f} m")
    print(f"  Median: {np.median(distances):.3f} m")
    print(f"  Max:    {distances.max():.3f} m")
    print(f"  Std:    {distances.std():.3f} m")

    # --- Histogram ---
    plt.figure(figsize=(6,4))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='k')
    plt.title(f"{name} distances to reference terrain")
    plt.xlabel("Distance (m)")
    plt.ylabel("Number of points")
    plt.show()

    # --- 3D Scatter (color-coded) ---
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(recon_cloud[:,0], recon_cloud[:,1], recon_cloud[:,2],
                    c=distances, cmap="jet", s=1)
    # Show terrain as gray
    ax.scatter(terrain_points[:,0], terrain_points[:,1], terrain_points[:,2],
               s=0.1, c="gray", alpha=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Distance to terrain (m)")
    ax.set_title(f"{name} Reconstruction vs Terrain (3D)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.show()

    # # --- 2D Projections (XY, XZ, YZ) ---
    # planes = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
    # fig, axes = plt.subplots(1, 3, figsize=(18,6))

    # for ax, (title, idx1, idx2) in zip(axes, planes):
    #     ax.scatter(terrain_points[:,idx1], terrain_points[:,idx2],
    #                s=0.1, c="gray", alpha=0.3, label="Terrain")
    #     sc2 = ax.scatter(recon_cloud[:,idx1], recon_cloud[:,idx2],
    #                      c=distances, cmap="jet", s=1)
    #     ax.set_title(f"{name} vs Terrain ({title} plane)")
    #     ax.set_xlabel(["X","X","Y"][planes.index((title, idx1, idx2))])
    #     ax.set_ylabel(["Y","Z","Z"][planes.index((title, idx1, idx2))])
    #     ax.axis("equal")
    # fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.6, label="Distance (m)")
    # plt.show()

    return distances

# ----------------------------
# Plot terrain ground truth alone
# ----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    terrain_points[:,0],
    terrain_points[:,1],
    terrain_points[:,2],
    c=terrain_points[:,2],    # colour mapped to Z (height)
    cmap="terrain",           # "terrain" colormap looks natural, but you can try "viridis", "plasma", etc.
    s=5,
    alpha=0.9
)

ax.set_title("Reference Terrain (Coloured by Height)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# add colorbar to show height mapping
fig.colorbar(sc, ax=ax, label="Height (Z)")

plt.show()

# ----------------------------
# Control flag
# ----------------------------
comparison = False  # Set to False to skip
if comparison:
    compare_and_visualize_colored(X_orb, terrain_points, "ORB")
    compare_and_visualize_colored(X_sift, terrain_points, "SIFT")


import random
from sklearn.linear_model import LinearRegression
# ----------------------------
# Enhanced random region analysis with image properties
# ----------------------------

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

def show_best_worst_regions(img_rgb, regions, method="SIFT"):
    """
    Display regions of interest with keypoints overlaid using normal-sized points.

    Parameters
    ----------
    img_rgb : np.ndarray
        Original RGB image
    regions : list of dict
        Each dict has keys: x1, y1, x2, y2, keypoints, etc.
    method : str
        Feature detector name ("SIFT" or "ORB")
    """
    detector = cv2.SIFT_create() if method=="SIFT" else cv2.ORB_create(nfeatures=500)
    n = len(regions)
    plt.figure(figsize=(5*n,5))
    
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        crop = img_rgb[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        kp = detector.detect(crop_gray, None)
        
        # Draw normal-sized keypoints
        kp_img = cv2.drawKeypoints(crop, kp, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        
        plt.subplot(1,n,i+1)
        plt.imshow(kp_img)
        plt.title(f"{method} Region {i+1}\nKeypoints: {len(kp)}")
        plt.axis("off")
    
    plt.show()


def analyze_regions_with_metrics(imgL_rgb, imgR_rgb, method="SIFT", nfeatures=1000, n_samples=1000,
                                 region_size=(300,300)):
    """
    Analyze random regions for keypoints, brightness, contrast, and contour density,
    plot metrics vs keypoints with linear regression, and print R coefficients.
    """
    h, w = imgL_rgb.shape[:2]
    rw, rh = region_size
    imgL_gray = cv2.cvtColor(imgL_rgb, cv2.COLOR_RGB2GRAY)
    imgR_gray = cv2.cvtColor(imgR_rgb, cv2.COLOR_RGB2GRAY)

    detector = cv2.SIFT_create(nfeatures=nfeatures) if method=="SIFT" else cv2.ORB_create(nfeatures=nfeatures)
    results = []

    for _ in range(n_samples):
        x1 = random.randint(0, w - rw)
        y1 = random.randint(0, h - rh)
        x2, y2 = x1 + rw, y1 + rh

        cropL = imgL_gray[y1:y2, x1:x2]
        cropR = imgR_gray[y1:y2, x1:x2]

        # Keypoints
        kpL = detector.detect(cropL, None)
        kpR = detector.detect(cropR, None)
        n_points = len(kpL) + len(kpR)

        # Brightness and contrast
        mean_brightness = (cropL.mean() + cropR.mean()) / 2
        contrast = (cropL.std() + cropR.std()) / 2

        # Contours/texture using Canny edges
        edgesL = cv2.Canny(cropL, 50, 150)
        edgesR = cv2.Canny(cropR, 50, 150)
        n_edges = np.sum(edgesL>0) + np.sum(edgesR>0)

        results.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "keypoints": n_points,
            "brightness": mean_brightness,
            "contrast": contrast,
            "edges": n_edges
        })

    # Convert to arrays
    keypoints = np.array([r["keypoints"] for r in results])
    brightness = np.array([r["brightness"] for r in results])
    contrast = np.array([r["contrast"] for r in results])
    edges = np.array([r["edges"] for r in results])

    # ----------------------------
    # Scatter plots with regression and R coefficient
    # ----------------------------
    def scatter_with_fit(x, y, xlabel, ylabel="Keypoints", color='blue'):
        plt.figure(figsize=(6,4))
        plt.scatter(x, y, color=color, alpha=0.6)
        # Fit line
        model = LinearRegression().fit(x.reshape(-1,1), y)
        y_fit = model.predict(x.reshape(-1,1))
        plt.plot(x, y_fit, color='red', linewidth=2)
        # Pearson R
        r, _ = pearsonr(x, y)
        print(f"{method} - {ylabel} vs {xlabel} R = {r:.3f}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{method}: {ylabel} vs {xlabel}")
        plt.grid(True)
    

    scatter_with_fit(edges, keypoints, "Number of Edge Pixels", color='blue')
    scatter_with_fit(brightness, keypoints, "Mean Brightness", color='orange')
    scatter_with_fit(contrast, keypoints, "Contrast (std dev)", color='green')
    plt.show()
    
    # Identify best/worst regions
    sorted_results = sorted(results, key=lambda r: r["keypoints"], reverse=True)
    best_regions = sorted_results[:3]
    worst_regions = sorted_results[-3:]

    return results, best_regions, worst_regions

results_orb, best_orb, worst_orb = analyze_regions_with_metrics(imgL_rgb, imgR_rgb, method="ORB",
                                                               nfeatures=1000, n_samples=1000,
                                                               region_size=(300,300))

results_sift, best_sift, worst_sift = analyze_regions_with_metrics(imgL_rgb, imgR_rgb, method="SIFT",
                                                                   nfeatures=1000, n_samples=1000,
                                                                   region_size=(300,300))

# ----------------------------
# Display best and worst regions for ORB
# ----------------------------
print("ORB - Best Regions")
show_best_worst_regions(imgL_rgb, best_orb, method="ORB")

print("ORB - Worst Regions")
show_best_worst_regions(imgL_rgb, worst_orb, method="ORB")

# ----------------------------
# Display best and worst regions for SIFT
# ----------------------------
print("SIFT - Best Regions")
show_best_worst_regions(imgL_rgb, best_sift, method="SIFT")

print("SIFT - Worst Regions")
show_best_worst_regions(imgL_rgb, worst_sift, method="SIFT")
