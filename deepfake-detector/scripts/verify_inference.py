import sys
import random
from pathlib import Path
import logging
import torch

# Ensure project root is on sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.detector import DeepfakeDetector

def verify_inference():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    
    # Path to the model
    checkpoint_path = _PROJECT_ROOT / "checkpoints" / "best.pth"
    if not checkpoint_path.exists():
        # Fallback to verify_run if best.pth not in root
        checkpoint_path = _PROJECT_ROOT / "checkpoints" / "verify_run" / "best.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path}...")
    try:
        detector = DeepfakeDetector(
            model_name="efficientnet_b0",
            weights_path=checkpoint_path,
            lazy_load=False
        )
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return
    
    test_dir = _PROJECT_ROOT / "dataset" / "test"
    categories = ["real", "fake"]
    
    results = []
    samples_per_cat = 50
    
    print(f"\n--- Running Verification (Sampling {samples_per_cat} per category) ---")
    
    for cat in categories:
        img_dir = test_dir / cat
        if not img_dir.exists():
            print(f"Warning: Directory {img_dir} does not exist.")
            continue
            
        all_images = [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        if not all_images:
            print(f"Warning: No images found in {img_dir}")
            continue
            
        random.seed(42)
        sampled_images = random.sample(all_images, min(len(all_images), samples_per_cat))
        
        for img_path in sampled_images:
            try:
                label, confidence = detector.predict(str(img_path))
                results.append({
                    "actual": cat.capitalize(),
                    "predicted": label,
                    "confidence": confidence,
                    "file": img_path.name
                })
            except Exception as e:
                print(f"Error predicting {img_path.name}: {e}")

    # Calculations
    total = len(results)
    correct = sum(1 for r in results if r["actual"] == r["predicted"])
    accuracy = correct / total if total > 0 else 0
    
    # Confusion Matrix (Actual rows, Predicted columns)
    # [ [Real-Real, Real-Fake], [Fake-Real, Fake-Fake] ]
    matrix = {"Real": {"Real": 0, "Fake": 0}, "Fake": {"Real": 0, "Fake": 0}}
    for r in results:
        matrix[r["actual"]][r["predicted"]] += 1
        
    # Generate report
    report_lines = []
    report_lines.append("DEEPFAKE DETECTOR VERIFICATION REPORT")
    report_lines.append("=====================================")
    report_lines.append(f"Model: {detector.model_name}")
    report_lines.append(f"Weights: {checkpoint_path}")
    report_lines.append(f"Total Samples: {total}")
    report_lines.append(f"Accuracy: {accuracy:.2%}")
    report_lines.append("")
    report_lines.append("Confusion Matrix:")
    report_lines.append(f"{'':<12} | {'Pred Real':<10} | {'Pred Fake':<10}")
    report_lines.append("-" * 38)
    report_lines.append(f"{'Actual Real':<12} | {matrix['Real']['Real']:<10} | {matrix['Real']['Fake']:<10}")
    report_lines.append(f"{'Actual Fake':<12} | {matrix['Fake']['Real']:<10} | {matrix['Fake']['Fake']:<10}")
    report_lines.append("")
    report_lines.append("Example Predictions:")
    report_lines.append(f"{'Actual':<10} | {'Predicted':<10} | {'Conf':<8} | {'File'}")
    report_lines.append("-" * 60)
    for r in results[:20]: # Show first 20
        report_lines.append(f"{r['actual']:<10} | {r['predicted']:<10} | {r['confidence']:>6.2f}% | {r['file']}")

    report_content = "\n".join(report_lines)
    print("\n" + report_content)
    
    report_path = _PROJECT_ROOT / "reports" / "verification_report.txt"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report_content)
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    verify_inference()
