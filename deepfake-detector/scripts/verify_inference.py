import sys
from pathlib import Path
import logging

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.detector import DeepfakeDetector

def verify_inference():
    # Setup logging to console only for this script
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Path to the newly trained model
    checkpoint_path = _PROJECT_ROOT / "checkpoints" / "verify_run" / "best.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    detector = DeepfakeDetector(
        model_name="efficientnet_b0",
        weights_path=checkpoint_path,
        lazy_load=False
    )
    
    test_dir = _PROJECT_ROOT / "dataset" / "test"
    categories = ["real", "fake"]
    
    results = []
    
    print("\n--- Inference Results ---")
    print(f"{'Category':<10} | {'File':<30} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 70)
    
    for cat in categories:
        img_dir = test_dir / cat
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                try:
                    label, confidence = detector.predict(str(img_path))
                    results.append({
                        "category": cat,
                        "file": img_path.name,
                        "prediction": label,
                        "confidence": confidence
                    })
                    print(f"{cat:<10} | {img_path.name[:30]:<30} | {label:<10} | {confidence:>9.2f}%")
                except Exception as e:
                    print(f"{cat:<10} | {img_path.name[:30]:<30} | Error: {e}")

    # Summary
    correct = sum(1 for r in results if r["category"].capitalize() == r["prediction"])
    accuracy = correct / len(results) if results else 0
    print("-" * 70)
    print(f"Total Tested: {len(results)}")
    print(f"Correct:      {correct}")
    print(f"Accuracy:     {accuracy:.2%}")

if __name__ == "__main__":
    verify_inference()
