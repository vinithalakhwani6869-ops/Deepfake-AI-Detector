import os
import shutil
import urllib.request
from pathlib import Path

def populate_dataset():
    dataset_root = Path("dataset")
    fixtures_root = Path("tests/fixtures")
    
    splits = ["train", "val", "test"]
    categories = ["real", "fake"]
    
    # 1. Use fixtures as base
    real_fixture = fixtures_root / "real_sample.jpg"
    fake_fixture = fixtures_root / "fake_sample.jpg"
    
    for split in splits:
        for cat in categories:
            dest_dir = dataset_root / split / cat
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            source = real_fixture if cat == "real" else fake_fixture
            if source.exists():
                shutil.copy(source, dest_dir / f"fixture_{split}_{cat}.jpg")
                print(f"Copied {source} to {dest_dir}")
            else:
                print(f"Warning: Fixture {source} not found.")

    # 2. Download additional "real" and "fake" samples from stable URLs
    # Real face from OpenCV
    real_urls = [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/face.png",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/group.jpg"
    ]
    # "Fake" (Synthetic) samples from StyleGAN samples in various repos
    fake_urls = [
        "https://raw.githubusercontent.com/NVlabs/stylegan/master/docs/stylegan-teaser.png",
        "https://raw.githubusercontent.com/deepfakes/faceswap/master/tests/lib/gui/img/style_test_1.png"
    ]
    
    for split in splits:
        # Real
        for i, url in enumerate(real_urls):
            dest = dataset_root / split / "real" / f"downloaded_real_{i}.jpg"
            print(f"Downloading {url} to {dest}...")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"Failed: {e}")
        # Fake
        for i, url in enumerate(fake_urls):
            dest = dataset_root / split / "fake" / f"downloaded_fake_{i}.png"
            print(f"Downloading {url} to {dest}...")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"Failed: {e}")

if __name__ == "__main__":
    populate_dataset()
