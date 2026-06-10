import os
import shutil
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_cifake():
    source_root = Path("CIFAKE-Real-and-AI-Generated-Synthetic-Images-main/DATASET")
    target_root = Path("dataset")
    
    # 1. Check if dataset already exists and is valid
    if target_root.exists():
        logger.info(f"Checking existing dataset at {target_root}")
        valid = True
        for split in ["train", "val", "test"]:
            for cat in ["real", "fake"]:
                p = target_root / split / cat
                if not p.exists() or len(list(p.glob("*.jpg"))) == 0:
                    valid = False
                    break
        if valid:
            logger.info("Dataset already exists and appears valid. Skipping setup.")
            print_stats(target_root)
            return

    # 2. Download dataset if source is missing
    if not source_root.exists():
        logger.info("Source dataset not found. Attempting to download...")
        try:
            import kagglehub
            path = kagglehub.dataset_download("prakaslanisetti/cifake-real-and-ai-generated-synthetic-images")
            # The downloaded path usually contains the files directly or in a subfolder
            downloaded_path = Path(path)
            logger.info(f"Downloaded dataset to {downloaded_path}")
            # Map source_root to the downloaded path
            if (downloaded_path / "train").exists():
                source_root = downloaded_path
            elif (downloaded_path / "train").exists(): # Check if it's one level deeper
                source_root = downloaded_path
            else:
                # Find where 'train' and 'test' are
                train_dirs = list(downloaded_path.rglob("train"))
                if train_dirs:
                    source_root = train_dirs[0].parent
                else:
                    raise FileNotFoundError("Could not find 'train' directory in downloaded dataset")
        except ImportError:
            logger.error("kagglehub not installed. Please install it with 'pip install kagglehub'")
            return
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return

    logger.info(f"Setting up dataset from {source_root} to {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    
    categories = {"FAKE": "fake", "REAL": "real"}
    val_split = 0.1 
    
    # 3. Process Train -> Train/Val
    for src_cat, tgt_cat in categories.items():
        src_train_dir = source_root / "train" / src_cat
        if not src_train_dir.exists():
            # Try lowercase
            src_train_dir = source_root / "train" / src_cat.lower()
            if not src_train_dir.exists():
                logger.error(f"Source directory {src_train_dir} does not exist.")
                continue
        
        images = list(src_train_dir.glob("*.jpg"))
        random.seed(42)
        random.shuffle(images)
        
        num_val = int(len(images) * val_split)
        val_images = images[:num_val]
        train_images = images[num_val:]
        
        (target_root / "train" / tgt_cat).mkdir(parents=True, exist_ok=True)
        (target_root / "val" / tgt_cat).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Copying {len(train_images)} images to train/{tgt_cat}...")
        for img in train_images:
            dest = target_root / "train" / tgt_cat / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
            
        logger.info(f"Copying {len(val_images)} images to val/{tgt_cat}...")
        for img in val_images:
            dest = target_root / "val" / tgt_cat / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
            
    # 4. Process Test
    for src_cat, tgt_cat in categories.items():
        src_test_dir = source_root / "test" / src_cat
        if not src_test_dir.exists():
            src_test_dir = source_root / "test" / src_cat.lower()
            if not src_test_dir.exists():
                logger.error(f"Source directory {src_test_dir} does not exist.")
                continue
                
        images = list(src_test_dir.glob("*.jpg"))
        (target_root / "test" / tgt_cat).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Copying {len(images)} images to test/{tgt_cat}...")
        for img in images:
            dest = target_root / "test" / tgt_cat / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

    print_stats(target_root)

def print_stats(target_root):
    logger.info("--- Dataset Statistics ---")
    for split in ["train", "val", "test"]:
        split_total = 0
        stats_str = f"{split.upper()}: "
        for tgt_cat in ["real", "fake"]:
            p = target_root / split / tgt_cat
            count = len(list(p.glob("*.jpg"))) if p.exists() else 0
            stats_str += f"{tgt_cat}: {count}  "
            split_total += count
        logger.info(f"{stats_str} Total: {split_total}")

if __name__ == "__main__":
    setup_cifake()
