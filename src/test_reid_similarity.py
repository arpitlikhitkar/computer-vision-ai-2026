"""
Offline Person Re-ID Similarity Diagnostic Script (Phase 4 Diagnostic)

ISOLATED TEST:
- NO YOLO
- NO ByteTrack
- NO Live Webcam
- Pure Person Re-ID Feature Extractor Diagnosis

MODEL SPECIFICATION:
- Architecture: OSNet (Omni-Scale Network, osnet_x0_25)
- Source: KaiyangZhou/deep-person-reid (Torchreid)
- Task: Dedicated Person Re-Identification (Re-ID Metric Learning)
- Embedding Dimension: 512-d (L2-Normalized Float Vector)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision.transforms as T
import cv2

try:
    import torchreid
except ImportError:
    print("[ERROR] torchreid is not installed in current environment.")
    sys.exit(1)


class OSNetReIDExtractor:
    """
    Dedicated Person Re-ID Feature Extractor using Torchreid OSNet.
    Produces L2-normalized feature embeddings.
    """
    def __init__(self, device="cpu"):
        import torchreid.reid.utils as utils
        self.extractor = utils.FeatureExtractor(
            model_name="osnet_x0_25",
            device=device,
            verbose=False
        )

    def extract_from_bgr(self, img_bgr):
        """Extracts L2-normalized feature embedding from BGR image array."""
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Invalid or empty image array passed to extract_from_bgr.")

        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # torchreid FeatureExtractor expects a list of PIL Images or numpy arrays
        features = self.extractor(img_rgb)
        
        # Flatten and L2 normalize
        feat = features.cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    def extract_from_file(self, img_path):
        """Loads image file and extracts 512-d L2-normalized feature embedding."""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"OpenCV failed to read image file: {img_path}")
        return self.extract_from_bgr(img_bgr)


def cosine_similarity(u, v):
    """Computes Cosine Similarity between two L2-normalized vectors."""
    # For L2-unit vectors (||u||=1, ||v||=1), cosine similarity is the dot product u . v
    return float(np.dot(u, v))


def run_diagnostic(img_a_path=None, img_b_path=None):
    print("==================================================")
    print(" PERSON RE-ID OFFLINE DIAGNOSTIC TOOL")
    print("==================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = OSNetReIDExtractor(device=device)

    print(f"Model Architecture: OSNet (osnet_x0_25)")
    print(f"Model Source:       KaiyangZhou/deep-person-reid")
    print(f"Preprocessing:      Resize (256, 128) -> RGB -> ImageNet Norm")
    print(f"Embedding Dim:      512")
    print(f"Device:             {device}")
    print("==================================================")

    # Use provided images or generate synthetic test images if paths not provided
    created_temp_files = False
    if img_a_path is None or img_b_path is None or not os.path.exists(str(img_a_path)):
        print("[INFO] No external images provided. Using crops from outputs/persons/ or generating test crops...")
        
        # Look for existing crops in outputs/persons/
        found_crops = []
        persons_dir = "outputs/persons"
        if os.path.exists(persons_dir):
            for pdir in sorted(os.listdir(persons_dir)):
                cdir = os.path.join(persons_dir, pdir, "crops")
                if os.path.exists(cdir):
                    for f in sorted(os.listdir(cdir)):
                        if f.endswith((".jpg", ".png")):
                            found_crops.append(os.path.join(cdir, f))

        if len(found_crops) >= 2:
            img_a_path = found_crops[0]
            img_b_path = found_crops[-1]
            print(f"[FOUND CROPS] Image A: {img_a_path}")
            print(f"[FOUND CROPS] Image B: {img_b_path}")
        else:
            # Create synthetic distinct person test crops for isolated model verification
            os.makedirs("scratch", exist_ok=True)
            img_a_path = "scratch/temp_person_a.jpg"
            img_b_path = "scratch/temp_person_b.jpg"
            created_temp_files = True

            # Person A: Blue shirt pattern
            crop_a = np.zeros((256, 128, 3), dtype=np.uint8)
            crop_a[40:180, 20:108] = [220, 100, 50]  # BGR Blue torso
            crop_a[0:40, 40:88] = [180, 180, 200]    # Skin/Face
            cv2.imwrite(img_a_path, crop_a)

            # Person B: Red shirt pattern
            crop_b = np.zeros((256, 128, 3), dtype=np.uint8)
            crop_b[40:180, 20:108] = [50, 50, 220]   # BGR Red torso
            crop_b[0:40, 40:88] = [120, 140, 160]   # Skin/Face
            cv2.imwrite(img_b_path, crop_b)

    # Extract Embeddings
    emb_a = extractor.extract_from_file(img_a_path)
    emb_b = extractor.extract_from_file(img_b_path)

    norm_a = float(np.linalg.norm(emb_a))
    norm_b = float(np.linalg.norm(emb_b))

    print()
    print("--- EMBEDDING VERIFICATION ---")
    print(f"Embedding A shape: {emb_a.shape} | L2 Norm: {norm_a:.4f}")
    print(f"Embedding B shape: {emb_b.shape} | L2 Norm: {norm_b:.4f}")
    print(f"Embedding A (first 10): {np.round(emb_a[:10], 4).tolist()}")
    print(f"Embedding B (first 10): {np.round(emb_b[:10], 4).tolist()}")
    
    are_different = not np.allclose(emb_a, emb_b)
    print(f"Are embeddings distinct? {are_different}")

    # Compute Similarities
    sim_a_a = cosine_similarity(emb_a, emb_a)
    sim_b_b = cosine_similarity(emb_b, emb_b)
    sim_a_b = cosine_similarity(emb_a, emb_b)

    print()
    print("--- SIMILARITY COMPARISON ---")
    print(f"Self-Similarity (Image A vs Image A): {sim_a_a:.4f}")
    print(f"Self-Similarity (Image B vs Image B): {sim_b_b:.4f}")
    print(f"Cross-Similarity (Image A vs Image B): {sim_a_b:.4f}")
    print("--------------------------------------------------")

    # Evaluate Embedding Collapse
    if sim_a_b > 0.92:
        print("[WARNING] High cross-similarity detected. Check crop variation.")
    else:
        print("[SUCCESS] Feature embeddings demonstrate clear spatial distinction!")

    # 5-Person Matrix Test (Requirement 6)
    run_matrix_test(extractor)

    return {
        "model": "OSNet (osnet_x0_25)",
        "embedding_dim": len(emb_a),
        "norm_a": norm_a,
        "norm_b": norm_b,
        "sim_a_a": sim_a_a,
        "sim_b_b": sim_b_b,
        "sim_a_b": sim_a_b,
        "distinct": are_different
    }


def run_matrix_test(extractor):
    """Generates 5 distinct synthetic/sample person crops and prints a 5x5 Similarity Matrix."""
    print()
    print("==================================================")
    print(" 5-PERSON EMBEDDING COLLAPSE DIAGNOSTIC MATRIX")
    print("==================================================")

    os.makedirs("scratch", exist_ok=True)
    labels = ["Person A", "Person B", "Person C", "Person D", "Person E"]
    colors = [
        [255, 0, 0],    # Blue
        [0, 255, 0],    # Green
        [0, 0, 255],    # Red
        [255, 255, 0],  # Cyan
        [0, 255, 255]   # Yellow
    ]
    
    embeddings = []
    for i, col in enumerate(colors):
        crop = np.zeros((256, 128, 3), dtype=np.uint8)
        crop[30:200, 15:113] = col  # Torso color pattern
        crop[0:30, 40:88] = [150, 150, 150]
        emb = extractor.extract_from_bgr(crop)
        embeddings.append(emb)

    n = len(labels)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    # Format header
    header_str = "          " + "".join([f"{lbl:>10}" for lbl in labels])
    print(header_str)
    print("-" * len(header_str))

    for i in range(n):
        row_str = f"{labels[i]:<10}" + "".join([f"{matrix[i, j]:10.4f}" for j in range(n)])
        print(row_str)

    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-ID Similarity Diagnostic Tool")
    parser.add_argument("img_a", nargs="?", default=None, help="Path to first person image")
    parser.add_argument("img_b", nargs="?", default=None, help="Path to second person image")
    args = parser.parse_args()

    run_diagnostic(args.img_a, args.img_b)
