import os
import urllib.request
from pathlib import Path

def download_images():
    # Corrected base URL
    base_url = "https://github.com/vzhou842/deepfake-detector/raw/master/data"
    
    dataset_root = Path("dataset")
    categories = ["real", "fake"]
    
    # We'll download 10 from train and 10 from test for each category
    sources = ["train", "test"]
    
    total_downloaded = 0
    
    for src in sources:
        for cat in categories:
            # We'll map 'train' source to 'train' split, and 'test' source to 'test' and 'val' splits
            if src == "train":
                dest_splits = ["train"]
            else:
                dest_splits = ["val", "test"]
                
            for split in dest_splits:
                dest_dir = dataset_root / split / cat
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # The repo has 0.jpg, 1.jpg, ... up to 9.jpg in both train and test
                for i in range(10):
                    filename = f"{i}.jpg"
                    url = f"{base_url}/{src}/{cat}/{filename}"
                    dest_path = dest_dir / f"dfdc_{src}_{cat}_{i}.jpg"
                    
                    if dest_path.exists():
                        continue
                        
                    print(f"Downloading {url} to {dest_path}...")
                    try:
                        urllib.request.urlretrieve(url, dest_path)
                        total_downloaded += 1
                    except Exception as e:
                        print(f"Failed to download {url}: {e}")
                    
    print(f"\nExpansion complete. Downloaded {total_downloaded} new images.")

if __name__ == "__main__":
    download_images()
