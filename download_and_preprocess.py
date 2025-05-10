#!/usr/bin/env python3
import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# download quickdraw numpy data for select categories
# downloaded all categories with [...] : quickdraw docs
# remember to start from 10 categories and keep bumping it up to by incremements of like 10ish? 
# all 350ish categories might not really be feasible to train on my gpu
# trial and error
# https://quickdraw.withgoogle.com/data

DEFAULT_CATEGORIES = ["cloud", "airplane", "butterfly", "door", "clock", "moon", "star", "mushroom", "light bulb", "flower"]
def download_data(categories, data_dir="quickdraw_data"):
    print(f"Downloading data to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    
    for category in categories:
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
        output_path = os.path.join(data_dir, f"{category}.npy")
        
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping download.")
            continue
        
        print(f"Downloading {category} dataset...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  
            file_size = int(response.headers.get('content-length', 0))
            
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

# script to convert .npy files to images (quickdraw data is stored as numpy arrays)
# and we need to convert them to images for training
# create test (20%) and train (80%) directories for each category and save images here
# also resize images to 224x224 for efficientnet input size (canvas input size is 28x28)

def npy_to_images(categories, data_dir="quickdraw_data", output_dir="quickdraw_images", samples_per_class=5000):

    print(f"Converting .npy files to images...")
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    try:
        resize_method = Image.Resampling.LANCZOS
    except AttributeError:
        resize_method = Image.LANCZOS
    
    for category in categories:
        print(f"Processing {category}...")
        try:
            npy_file = os.path.join(data_dir, f"{category}.npy")
            data = np.load(npy_file)
            
            if len(data) > samples_per_class:
                data = data[:samples_per_class]
            
            train_size = int(0.8 * len(data))
            train_data = data[:train_size]
            test_data = data[train_size:]

            train_category_dir = os.path.join(output_dir, "train", category)
            test_category_dir = os.path.join(output_dir, "test", category)
            os.makedirs(train_category_dir, exist_ok=True)
            os.makedirs(test_category_dir, exist_ok=True)
            
            for i, img_data in enumerate(tqdm(train_data, desc=f"Train {category}")):
                img = Image.fromarray(img_data.reshape(28, 28).astype(np.uint8))
                img = img.convert("RGB").resize((224, 224), resize_method)
                img.save(os.path.join(train_category_dir, f"{i}.png"))
            
            for i, img_data in enumerate(tqdm(test_data, desc=f"Test {category}")):
                img = Image.fromarray(img_data.reshape(28, 28).astype(np.uint8))
                img = img.convert("RGB").resize((224, 224), resize_method)
                img.save(os.path.join(test_category_dir, f"{i}.png"))
                
        except Exception as e:
            print(f"Error processing {category}: {str(e)}")
    
    print("Image conversion complete!")

def visualize_samples(categories, data_dir="quickdraw_data", num_samples=5):

    plt.figure(figsize=(10, len(categories)*2))
    
    for i, category in enumerate(categories):
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