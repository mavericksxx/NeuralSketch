#!/usr/bin/env python3
import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Categories to download
DEFAULT_CATEGORIES = ["cloud", "airplane", "butterfly", "door", "clock", "moon", "star", "mushroom", "light bulb", "flower"]
def download_data(categories, data_dir="quickdraw_data"):
    """
    Download the QuickDraw numpy bitmap data for the given categories
    """
    print(f"Downloading data to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    
    for category in categories:
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
        output_path = os.path.join(data_dir, f"{category}.npy")
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping download.")
            continue
        
        print(f"Downloading {category} dataset...")
        try:
            # Using requests instead of wget
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Get file size for progress bar
            file_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(output_path, 'wb') as f, tqdm(
                desc=category,
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            print(f"Failed to download {category} dataset: {str(e)}")

def npy_to_images(categories, data_dir="quickdraw_data", output_dir="quickdraw_images", samples_per_class=5000):
    """
    Convert the .npy files to individual image files
    """
    print(f"Converting .npy files to images...")
    # Create train and test directories for each category
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Handle PIL resizing method compatibility
    try:
        # For newer Pillow versions (9.0.0+)
        resize_method = Image.Resampling.LANCZOS
    except AttributeError:
        # For older Pillow versions
        resize_method = Image.LANCZOS
    
    for category in categories:
        print(f"Processing {category}...")
        try:
            # Load the numpy bitmap data
            npy_file = os.path.join(data_dir, f"{category}.npy")
            data = np.load(npy_file)
            
            # Take a subset of the data
            if len(data) > samples_per_class:
                data = data[:samples_per_class]
            
            # Split into train (80%) and test (20%) sets
            train_size = int(0.8 * len(data))
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Create category directories
            train_category_dir = os.path.join(output_dir, "train", category)
            test_category_dir = os.path.join(output_dir, "test", category)
            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(test_category_dir, exist_ok=True)
            
            # Save train images
            for i, img_data in enumerate(tqdm(train_data, desc=f"Train {category}")):
                img = Image.fromarray(img_data.reshape(28, 28).astype(np.uint8))
                # Convert to RGB and resize to 224x224 (EfficientNet input size)
                img = img.convert("RGB").resize((224, 224), resize_method)
                img.save(os.path.join(train_category_dir, f"{i}.png"))
            
            # Save test images
            for i, img_data in enumerate(tqdm(test_data, desc=f"Test {category}")):
                img = Image.fromarray(img_data.reshape(28, 28).astype(np.uint8))
                # Convert to RGB and resize to 224x224 (EfficientNet input size)
                img = img.convert("RGB").resize((224, 224), resize_method)
                img.save(os.path.join(test_category_dir, f"{i}.png"))
                
        except Exception as e:
            print(f"Error processing {category}: {str(e)}")
    
    print("Image conversion complete!")

def visualize_samples(categories, data_dir="quickdraw_data", num_samples=5):
    """
    Visualize a few samples from each category
    """
    plt.figure(figsize=(10, len(categories)*2))
    
    for i, category in enumerate(categories):
        # Load the numpy bitmap data
        npy_file = os.path.join(data_dir, f"{category}.npy")
        try:
            data = np.load(npy_file)
            for j in range(num_samples):
                plt.subplot(len(categories), num_samples, i*num_samples + j + 1)
                plt.imshow(data[j].reshape(28, 28), cmap="gray")
                if j == 0:
                    plt.ylabel(category)
                plt.xticks([])
                plt.yticks([])
        except Exception as e:
            print(f"Error visualizing {category}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig("quickdraw_samples.png")
    print("Sample visualization saved as 'quickdraw_samples.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess QuickDraw dataset")
    parser.add_argument("--categories", type=str, nargs="+", default=DEFAULT_CATEGORIES,
                        help=f"Categories to download (default: {DEFAULT_CATEGORIES})")
    parser.add_argument("--data_dir", type=str, default="quickdraw_data",
                        help="Directory to store the downloaded .npy files")
    parser.add_argument("--output_dir", type=str, default="quickdraw_images",
                        help="Directory to store the processed images")
    parser.add_argument("--samples_per_class", type=int, default=5000,
                        help="Number of samples to use per class (default: 5000)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize sample images from the dataset")
    
    args = parser.parse_args()
    
    download_data(args.categories, args.data_dir)
    
    if args.visualize:
        visualize_samples(args.categories, args.data_dir)
    
    npy_to_images(args.categories, args.data_dir, args.output_dir, args.samples_per_class)
    
    print("Data download and preprocessing complete!")