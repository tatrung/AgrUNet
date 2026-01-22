#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:26:57 2025

@author: tahoangtrung
"""
import os
import numpy as np
import tifffile as tiff 

import matplotlib.pyplot as plt
 
import shutil
import random
import re
from glob import glob


from collections import defaultdict


from tqdm import tqdm

import albumentations as A


def check_masks_for_zero(mask_dir):
    """
    Check all TIFF mask files in a directory for the presence of pixel value 0.

    Args:
        mask_dir (str): Directory containing mask TIFF files.

    Returns:
        List of filenames that contain value 0.
    """
    files_with_zero = []

    for f in sorted(glob(os.path.join(mask_dir, "*.tif"))):
        mask = tiff.imread(f)
        if mask.ndim == 3:
            mask = mask[..., 0]
        unique_vals = np.unique(mask)
        if 0 in unique_vals:
            print(f"❗️ {os.path.basename(f)} contains 0")
            files_with_zero.append(os.path.basename(f))

    print(f"\n✅ Total files with 0 value: {len(files_with_zero)}")
    return files_with_zero

def create_patches(
    image_dir, 
    mask_dir, 
    patch_image_dir, 
    patch_mask_dir,    
    stride, 
    num_classes, 
    start_index,
    patch_size=(128, 128)
):
    import numpy as np
    import os
    from tqdm import tqdm
    
    os.makedirs(patch_image_dir, exist_ok=True)
    os.makedirs(patch_mask_dir, exist_ok=True)

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    report = []
    patch_count = start_index
    skipped_count = 0

    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Patching"):
        image_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)
        
        # 🔄 Automatically fix orientation if channels come first (Bands, H, W)
        if image.ndim == 3 and image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
           image = np.transpose(image, (1, 2, 0))
        

        # Drop last channel if mask has 3D shape
        if mask.ndim == 3:
            mask = mask[..., 0]
            
        # ✅ Check shape alignment
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"❌ Shape mismatch in {img_file}:\n"
                f"  Image shape: {image.shape}\n"
                f"  Mask shape:  {mask.shape}"
            )

        # replace nan, +/-inf by value 0.0
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = image.astype(np.float32)

        height, width = mask.shape

        for i in range(0, height - patch_size[0] + 1, stride):
            for j in range(0, width - patch_size[1] + 1, stride):
                img_patch = image[i:i+patch_size[0], j:j+patch_size[1], :]
                mask_patch = mask[i:i+patch_size[0], j:j+patch_size[1]]

                # Force dtype
                img_patch = img_patch.astype(np.float32)

                # Warn if unexpected values
                has_nan = np.isnan(img_patch).any()
                has_posinf = np.isposinf(img_patch).any()
                has_neginf = np.isneginf(img_patch).any()
                correct_dtype = img_patch.dtype == np.float32
                if has_nan or has_posinf or has_neginf or not correct_dtype:
                    print(
                        f"⚠️ Warning in patch {patch_count} of file {img_file}:",
                        f"Contains NaN: {has_nan}, +Inf: {has_posinf}, -Inf: {has_neginf}, float32 dtype: {correct_dtype}"
                    )

                if np.all(mask_patch == 255):
                    skipped_count += 1
                    continue

                patch_name = f"patch_{patch_count:05d}"
                np.save(os.path.join(patch_image_dir, f"{patch_name}.npy"), img_patch)
                np.save(os.path.join(patch_mask_dir, f"{patch_name}.npy"), mask_patch)

                for category in range(num_classes):
                    pixel_count = np.sum(mask_patch == category)
                    report.append([patch_name, category, pixel_count])

                patch_count += 1

    print(f"✅ Done. Total patches created: {patch_count - start_index}, Skipped (NoData): {skipped_count}")

def find_zero_size_npy_files(folder_path):
    """
    Scan a folder for .npy files and print out files with zero-sized arrays.

    Args:
        folder_path (str): Path to the folder containing .npy files.
    """
    zero_files = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = np.load(file_path)
                if data.size == 0:
                    print(f"Zero-size array: {file_path}")
                    zero_files.append(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"\nTotal zero-size files: {len(zero_files)}")
    return zero_files

def scan_npy_mask_labels(mask_dir):
    # Scan all npy files then report the number of categories across the npy files.
    all_labels = set()
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.npy")))

    if not mask_paths:
        print("❌ No .npy files found in:", mask_dir)
        return

    print(f"🔍 Scanning {len(mask_paths)} mask files...\n")

    for path in mask_paths:
        mask = np.load(path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        unique_vals = np.unique(mask)
        all_labels.update(unique_vals)
        #print(f"{os.path.basename(path)} → labels: {unique_vals}")

    print("\n✅ Done scanning.")
    print(f"📊 Total unique labels found across all masks: {sorted(all_labels)}")

def list_npy_with_class_zero(mask_dir):
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.npy")))
    files_with_class_0 = []

    for path in tqdm(mask_paths, desc="Scanning masks"):
        mask = np.load(path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # squeeze 3D mask

        unique_values = np.unique(mask)
        if 0 in unique_values:
            files_with_class_0.append((os.path.basename(path), unique_values))

    print(f"\n🔍 Found {len(files_with_class_0)} mask files containing class 0:")
    for fname, unique_vals in files_with_class_0:
        print(f" - {fname}: {sorted(unique_vals)}")

    return files_with_class_0


def remap_npy_masks(input_mask_dir, output_mask_dir, ignore_label):
    """
    Remap class labels in .npy mask files to a continuous scheme starting from 1.
    The ignore label (e.g., 0) is kept unchanged.
    
    Args:
        input_mask_dir (str): Directory containing original .npy mask files.
        output_mask_dir (str): Directory to save remapped .npy masks.
        ignore_label (int): Label to be preserved and not remapped (e.g., 0 for background/ignore).
    
    Returns:
        label_map (dict): Mapping from original labels to new labels.
        reverse_map (dict): Mapping from new labels back to original labels.
    """
    
    os.makedirs(output_mask_dir, exist_ok=True)
    mask_paths = sorted(glob(os.path.join(input_mask_dir, "*.npy")))
    print(f"🔍 Found {len(mask_paths)} mask files to process.")

    # Step 1: Collect all unique labels (excluding ignore)
    unique_labels = set()
    for path in mask_paths:
        mask = np.load(path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        labels = np.unique(mask)
        labels = [l for l in labels if l != ignore_label]
        unique_labels.update(labels)

    sorted_labels = sorted(unique_labels)
    label_map = {orig: idx + 1 for idx, orig in enumerate(sorted_labels)}  # start from 1
    reverse_map = {v: k for k, v in label_map.items()}

    print(f"📊 Found {len(label_map)} unique valid labels (remapped to 1–{len(label_map)}), keeping {ignore_label} unchanged:")
    for orig, new in label_map.items():
        print(f"  {orig} → {new}")
    print(f"  {ignore_label} → {ignore_label} (ignored)")

    # Step 2: Remap masks
    for path in tqdm(mask_paths, desc="Remapping masks"):
        mask = np.load(path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        remapped = np.full_like(mask, fill_value=ignore_label)
        for orig_label, new_label in label_map.items():
            remapped[mask == orig_label] = new_label

        output_path = os.path.join(output_mask_dir, os.path.basename(path))
        np.save(output_path, remapped)

    print(f"\n✅ Remapped masks saved to: {output_mask_dir}")
    return label_map, reverse_map

def replace_zero_with_255_in_npy(folder_path):
    """
    Replaces all 0 values in .npy files with 255 and overwrites the original files.

    Parameters:
        folder_path (str): Path to the folder containing .npy files.
    """
    for fname in os.listdir(folder_path):
        if fname.endswith(".npy"):
            file_path = os.path.join(folder_path, fname)
            array = np.load(file_path)

            array[array == 0] = 255  # Replace 0 with 255
            np.save(file_path, array)

    print("Finish")
            
def replace_255_with_zero_in_npy(folder_path):
    """
    Replaces all 255 values in .npy files with 0 and overwrites the original files.

    Parameters:
        folder_path (str): Path to the folder containing .npy files.
    """
    for fname in os.listdir(folder_path):
        if fname.endswith(".npy"):
            file_path = os.path.join(folder_path, fname)
            array = np.load(file_path)

            array[array == 255] = 0  # Replace 0 with 255
            np.save(file_path, array)

            print(f"Processed: {fname}")

def filter_validation_patches(
    mask_dir,
    image_dir,
    output_mask_dir,
    output_image_dir,
    ignore_label,
    min_valid_pixel_ratio=0.1,
    min_unique_classes=1,
    verbose=True
):
    """
    Filters validation patches by excluding masks with too few valid pixels
    or insufficient class diversity.

    Parameters:
        mask_dir (str): Path to directory containing validation mask .npy files.
        image_dir (str): Path to directory containing corresponding validation image .npy files.
        output_mask_dir (str): Path to save filtered masks.
        output_image_dir (str): Path to save filtered images.
        ignore_label (int): Label to ignore (default: 255).
        min_valid_pixel_ratio (float): Minimum ratio of valid pixels (non-ignore) required.
        min_unique_classes (int): Minimum number of unique valid classes (excluding ignore).
        verbose (bool): Whether to print filtering summary.
    """
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    kept = 0
    discarded = 0

    for fname in os.listdir(mask_dir):
        if not fname.endswith('.npy'):
            continue

        mask_path = os.path.join(mask_dir, fname)
        mask = np.load(mask_path)

        # Valid pixels = not ignore label
        valid_mask = (mask != ignore_label)

        if np.sum(valid_mask) == 0:
            discarded += 1
            continue  # Skip this patch

        valid_pixel_ratio = np.sum(valid_mask) / mask.size

        # Get unique valid classes (excluding ignore_label only)
        valid_classes = np.unique(mask[valid_mask])

        if valid_pixel_ratio >= min_valid_pixel_ratio and len(valid_classes) >= min_unique_classes:
            # Keep this patch
            shutil.move(mask_path, os.path.join(output_mask_dir, fname))
            img_path = os.path.join(image_dir, fname)
            if os.path.exists(img_path):
                shutil.move(img_path, os.path.join(output_image_dir, fname))
            kept += 1
        else:
            discarded += 1

    if verbose:
        print(f"  Kept     : {kept}")
        print(f"  Discarded: {discarded}")

def move_matching_files(src1, src2, dst1, dst2, percentage):
    # Create destination folders if they don't exist
    os.makedirs(dst1, exist_ok=True)
    os.makedirs(dst2, exist_ok=True)

    # Get common filenames
    files1 = set(os.listdir(src1))
    files2 = set(os.listdir(src2))
    matching_files = sorted(files1.intersection(files2))

    # Shuffle and select the top N% of files
    num_to_move = int(len(matching_files) * (percentage / 100))
    selected_files = random.sample(matching_files, num_to_move)

    print(f"📁 Moving {num_to_move} matching file pairs ({percentage}%) from:")
    print(f"    {src1} → {dst1}")
    print(f"    {src2} → {dst2}")

    for fname in selected_files:
        shutil.move(os.path.join(src1, fname), os.path.join(dst1, fname))
        shutil.move(os.path.join(src2, fname), os.path.join(dst2, fname))

    print("✅ Done.")

def count_class_pixels(mask_folder, ignore_label):
    """
    Automatically find the max class label, and count total pixels per class from 0 up to max label,
    ignoring pixels with ignore_label and excluding classes with zero pixels.

    Returns:
        class_ids: array of class IDs that have non-zero pixels (0-based indexing)
        class_pixels: array of pixel counts per class
        class_percentages: array of percentage of each class
        total_valid: total number of valid (non-ignore) pixels
    """
    # First pass to find max class label (excluding ignore)
    max_label = 0
    print("🔍 Scanning for maximum class label...")
    for file in tqdm(os.listdir(mask_folder), desc="Scanning"):
        if not file.endswith(".npy"):
            continue
        mask = np.load(os.path.join(mask_folder, file))
        unique = np.unique(mask)
        unique = unique[unique != ignore_label]
        if len(unique) > 0:
            max_label = max(max_label, unique.max())

    total_pixels = np.zeros(max_label + 1, dtype=np.int64)
    total_valid = 0

    # Second pass: count pixels
    print("📊 Counting class pixels...")
    for file in tqdm(os.listdir(mask_folder), desc="Counting"):
        if not file.endswith(".npy"):
            continue
        mask = np.load(os.path.join(mask_folder, file))
        valid = mask != ignore_label
        total_valid += np.sum(valid)

        for cls in range(0, max_label + 1):
            total_pixels[cls] += np.sum((mask == cls) & valid)

    # Filter out classes with 0 pixels
    class_ids = np.arange(0, max_label + 1)
    nonzero_mask = total_pixels > 0
    class_ids = class_ids[nonzero_mask]
    class_pixels = total_pixels[nonzero_mask]
    class_percentages = (class_pixels / total_valid) * 100

    print("\n📊 Class Distribution:")
    for cls, count, percent in zip(class_ids, class_pixels, class_percentages):
        print(f"  Class {cls:2}: {count:8d} pixels ({percent:.2f}%)")

    return class_pixels

def compute_aug_factors(class_pixel_counts, base_factor=1.0, apply_percent=0.3):
    class_pixel_counts = np.array(class_pixel_counts, dtype=np.float32)

    # Skip class 0 (assumed to be background or unused)
    counts = np.array(class_pixel_counts, dtype=np.float32)

    max_pixels = np.max(counts)

    # Compute raw log-based factors
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_factors = np.log1p(max_pixels / counts) * base_factor
        raw_factors[np.isinf(raw_factors)] = 1
        raw_factors[np.isnan(raw_factors)] = 1
        raw_factors = np.clip(raw_factors, 1, None)

    # Default to 1.0 for all active classes (excluding index 0)
    final_factors = np.ones_like(raw_factors, dtype=np.float32)

    # Apply only to least represented X% classes
    num_classes_to_boost = max(1, int(len(counts) * apply_percent))
    least_classes = np.argsort(counts)[:num_classes_to_boost]
    final_factors[least_classes] = raw_factors[least_classes]

    return final_factors  # length = len(class_pixel_counts) - 1


def build_aug_plan(mask_folder, aug_factors, ignore_label, top_patch_percent):
    """
    Build augmentation plan based on per-class imbalance, selecting top patches
    with highest total under-represented class pixels.

    Args:
        mask_folder (str): Folder of .npy mask files.
        aug_factors (np.array): Per-active-class augmentation factors (len must match active classes).
        ignore_label (int): Label to ignore.
        top_patch_percent (float): Fraction of patches to augment (e.g., 0.4).

    Returns:
        aug_plan (list): (filename, factor) tuples
        total_aug_pixels (np.array): Per-class total pixel counts (after factor)
        active_class_indices (np.array): Active class labels
    """

    # Step 1: Discover all class labels across dataset
    print("🔍 Scanning mask files for class labels...")
    label_set = set()
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".npy")])
    for file in tqdm(mask_files, desc="Scanning"):
        mask = np.load(os.path.join(mask_folder, file))
        unique = np.unique(mask[mask != ignore_label])
        label_set.update(unique.tolist())

    active_class_labels = sorted([int(c) for c in label_set if c > 0])
    active_class_indices = np.array(active_class_labels)
    print(f"✅ Detected active class labels: {active_class_labels} | Ignore label: {ignore_label}" )

    if len(aug_factors) != len(active_class_labels):
        raise ValueError(f"aug_factors length {len(aug_factors)} ≠ {len(active_class_labels)} active classes")

    # Step 2: Count total pixels per class
    print("📊 Counting pixels per class...")
    class_totals = defaultdict(int)
    for file in tqdm(mask_files, desc="Counting"):
        mask = np.load(os.path.join(mask_folder, file))
        valid = mask != ignore_label
        for cls in active_class_labels:
            class_totals[cls] += np.sum((mask == cls) & valid)

    print(f"📊 Total pixels per class: {dict(class_totals)}")

    # Step 3: Score patches based on sum of underrepresented pixels
    patch_scores = []
    under_labels = [cls for i, cls in enumerate(active_class_labels) if aug_factors[i] > 1]
    print(f"🔎 Underrepresented classes to boost: {under_labels}")

    for file in tqdm(mask_files, desc="Scoring patches"):
        mask = np.load(os.path.join(mask_folder, file))
        valid = mask != ignore_label

        class_counts = np.array([np.sum((mask == cls) & valid) for cls in active_class_labels])
        under_score = sum(
            class_counts[i] if cls in under_labels else 0
            for i, cls in enumerate(active_class_labels)
        )
        patch_scores.append((file, under_score, class_counts))

    # Step 4: Select top patches
    patch_scores.sort(key=lambda x: x[1], reverse=True)
    top_n = int(len(patch_scores) * top_patch_percent)

    # Step 5: Build augmentation plan
    aug_plan = []
    total_aug_pixels = np.zeros(len(active_class_labels), dtype=np.int64)

    print(f"\n🚀 Building augmentation plan for top {top_patch_percent*100:.1f}% patches...")
    for i, (file, _, class_counts) in enumerate(patch_scores):
        if i < top_n:
            # Use max underrepresented class factor
            dominant_idx = np.argmax([
                class_counts[j] if active_class_labels[j] in under_labels else 0
                for j in range(len(active_class_labels))
            ])
            factor = int(np.clip(aug_factors[dominant_idx], 1, 10))
        else:
            factor = 1

        aug_plan.append((file, factor))
        total_aug_pixels += class_counts * factor

    # Step 6: Final report
    print("\n📊 Final weighted pixel distribution:")
    total = total_aug_pixels.sum()
    for i, cls in enumerate(active_class_labels):
        pct = 100 * total_aug_pixels[i] / total if total > 0 else 0
        print(f"  Class {cls:2d}: {total_aug_pixels[i]:7d} pixels ({pct:.2f}%)")

    return aug_plan, total_aug_pixels, active_class_indices

def augment_patches_with_plan(
    image_dir,
    mask_dir,
    aug_plan,
    out_image_dir,
    out_mask_dir,
    num_classes,
    ignore_label=255
):
    """
    Augments patches and saves them immediately to disk (low RAM).
    Then splits them into training and validation sets.

    Args:
        image_dir (str): Directory containing original image patches (.npy).
        mask_dir (str): Directory containing corresponding mask patches (.npy).
        aug_plan (list): List of tuples (filename, num_augmentations).
        train_image_dir (str): Output directory for training images.
        train_mask_dir (str): Output directory for training masks.
        val_image_dir (str): Output directory for validation images.
        val_mask_dir (str): Output directory for validation masks.
        test_size (float): Proportion of data to use for validation.
        num_classes (int): Number of valid classes (1–num_classes).
        ignore_label (int): Label to ignore (default: 255).
    """

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)


    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.7),
        #A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=ignore_label, p=0.7),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-10, 10), p=0.5),
      
        #A.OneOf([
        #    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.05, p=0.5),
        #    #A.RandomGamma(gamma_limit=(10, 40), p=1.0),
        #], p=0.5),
        #A.GaussNoise(std_range=(0.02,0.1),mean_range=(-0.01,0.01), p =0.3),
        #A.ElasticTransform(alpha=1.0, sigma=50.0, alpha_affine=30.0, p=0.4)
        #A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(0,8), hole_width_range=(0,8)),
        
    ],
    additional_targets={'mask': 'mask'})  # ensure mask is treated as discrete
    #)
    print("🔁 Augmenting and saving to disk...")
    for file_name, factor in tqdm(aug_plan, desc="Augmenting"):
        base_name = os.path.splitext(file_name)[0]
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"⚠️ Skipping missing pair: {file_name}")
            continue

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        mask = mask.astype(np.uint8)  # enforce integer type

        for i in range(factor):
            augmented = augmentation(image=image, mask=mask)
            aug_image = augmented["image"]
            aug_mask = augmented["mask"]     
            # Fix NaN and Inf in augmented image
            aug_image = np.nan_to_num(aug_image, nan=0.0, posinf=1e6, neginf=-1e6)
            aug_mask = np.nan_to_num(aug_mask, nan=0.0, posinf=1e6, neginf=-1e6)

           
         # Clip mask to valid class range and keep ignore label
            aug_mask = np.clip(aug_mask, 0, max(num_classes, ignore_label)).astype(np.uint8)

            aug_filename = f"{base_name}_aug{i}.npy"
            np.save(os.path.join(out_image_dir, aug_filename), aug_image)
            np.save(os.path.join(out_mask_dir, aug_filename), aug_mask)

    print("✅ All augmentations saved to disk.")

def move_npy_files(start_num, end_num, input_folder, output_folder):
    """
    Moves .npy files based on the numeric ID at the end of the filename
    (e.g., '..._00001.npy') from input_folder to output_folder.
    
    Args:
        start_num (int): Starting number (e.g., 1).
        end_num (int): Ending number (e.g., 100).
        input_folder (str): Source folder path.
        output_folder (str): Destination folder path.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Regex to match files ending with '_00001.npy' to '_10000.npy'
    pattern = re.compile(r'^(.*_)(\d{5})\.npy$')

    for filename in os.listdir(input_folder):
        match = pattern.match(filename)
        if match:
            num = int(match.group(2))
            if start_num <= num <= end_num:
                src_path = os.path.join(input_folder, filename)
                dst_path = os.path.join(output_folder, filename)
                shutil.move(src_path, dst_path)
                print(f"Moved: {filename}")

def check_image_and_mask(image_path, mask_path):
    print(f"\n=== Checking image: {image_path} ===")
    image = np.load(image_path)
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image min/max: {image.min()} / {image.max()}")
    
    print(f"\n=== Checking mask: {mask_path} ===")
    mask = np.load(mask_path)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    unique_vals = np.unique(mask)
    print(f"Unique values in mask: {unique_vals}")

    # Check for one-hot encoding
    if len(mask.shape) == 3 and mask.shape[-1] > 1:
        print("⚠️ Detected one-hot encoding (shape: H x W x C).")
        class_map = np.argmax(mask, axis=-1)
        print(f"Class map (argmax) shape: {class_map.shape}")
        print(f"Unique classes (argmax): {np.unique(class_map)}")
        mask_for_stats = class_map
    else:
        print("✅ Detected sparse mask (shape: H x W).")
        mask_for_stats = mask

    # Check for ignore label
    if 255 in unique_vals:
        print("⚠️ IGNORE_LABEL (255) is present in the mask.")

    # Class distribution (excluding 255)
    valid_pixels = mask_for_stats[mask_for_stats != 255]
    if valid_pixels.size > 0:
        classes, counts = np.unique(valid_pixels, return_counts=True)
        print("\nClass distribution (excluding 255):")
        for cls, count in zip(classes, counts):
            print(f"  Class {cls}: {count} pixels")
    else:
        print("❌ No valid (non-255) pixels found in mask.")

def inspect_npy_patch(image_path, mask_path, ignore_label=0, max_classes=10):
    """
    Load and display an image patch and its corresponding mask from .npy files.

    Parameters:
        image_path (str): Path to the .npy image file.
        mask_path (str): Path to the .npy mask file.
        ignore_label (int): Value used to denote ignored pixels in the mask.
        max_classes (int): Max number of classes expected in the mask for visualization.
    """

    image = np.load(image_path)  # shape (H, W, 10)
    mask = np.load(mask_path)

    print(f"Image shape: {image.shape}, dtype: {image.dtype}, min: {np.min(image)}, max: {np.max(image)}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)}")

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # Select Sentinel-2 true color bands: Red=band4(3), Green=band3(2), Blue=band2(1)
    if image.shape[-1] >= 4:
        rgb = image[..., [3, 2, 1]]
    else:
        # fallback if fewer bands
        rgb = image[..., :3]

    # Normalize rgb for display
    img_min, img_max = rgb.min(), rgb.max()
    if img_max > img_min:
        rgb = (rgb - img_min) / (img_max - img_min)
    else:
        rgb = np.zeros_like(rgb)

    # Prepare mask for visualization
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask_display = np.argmax(mask, axis=-1)
    else:
        mask_display = mask

    mask_display_viz = np.copy(mask_display)
    if ignore_label in np.unique(mask_display):
        mask_display_viz[mask_display == ignore_label] = max_classes

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("Sentinel-2 True Color Composite (R4,G3,B2)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    cmap = plt.get_cmap('tab10', max_classes + 1)
    plt.imshow(mask_display_viz, cmap=cmap, vmin=0, vmax=max_classes)
    plt.title("Mask (labels)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def check_nan(image):
    if np.isnan(image).any():
        print("The image contains NaN values.")
        return True
    else:
        print("The image does NOT contain any NaN values.")
        return False

def check_invalid_values_in_npy_folder(folder_path):
    """
    Scans all .npy files in the folder and reports files containing NaN or Inf values.
    """
    invalid_files = []
    total_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            total_files += 1
            file_path = os.path.join(folder_path, filename)
            try:
                data = np.load(file_path)
                if np.isnan(data).any() or np.isinf(data).any():
                    invalid_files.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                invalid_files.append(filename)

    print(f"\nChecked {total_files} .npy files in: {folder_path}")
    if invalid_files:
        print(f"❌ {len(invalid_files)} files contain NaN or Inf values:")
        for f in invalid_files:
            print(" -", f)
    else:
        print("✅ All files are clean (no NaN or Inf values).")

    return invalid_files

def plot_and_save_npy_samples(
    train_img_dir, train_mask_dir,
    val_img_dir, val_mask_dir,
    output_dir="sample_plots",
    num_samples=5,
    rgb_indices=(3, 2, 1),
    ignore_label=255,
    plot_pixel_values=True,              # Toggle value display
    num_pixels_per_class=2              # Number of pixels to annotate per class
):
    os.makedirs(output_dir, exist_ok=True)

    def overlay_class_labels(ax, mask, max_per_class, show_values):
        """Overlay class label values at random pixels on the mask."""
        mask = mask.squeeze()
        unique_classes = np.unique(mask)
        class_counts = {cls: 0 for cls in unique_classes if cls != ignore_label}
    
        for cls in class_counts.keys():
            positions = np.argwhere(mask == cls)
            if len(positions) == 0:
                continue
            selected = positions[np.random.choice(len(positions), min(max_per_class, len(positions)), replace=False)]
            selected = selected.reshape(-1, 2)  # Ensure 2D coords
            for yx in selected:
                y, x = yx
                if show_values:
                    ax.text(
                        x, y, str(cls),
                        color='white', fontsize=6,
                        ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
                    )


    def plot_samples(img_dir, mask_dir, prefix):
        all_img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
        all_mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

        indices = random.sample(range(min(len(all_img_files), len(all_mask_files))), min(num_samples, len(all_img_files)))
        selected_img_files = [all_img_files[i] for i in indices]
        selected_mask_files = [all_mask_files[i] for i in indices]

        for i, (img_file, mask_file) in enumerate(zip(selected_img_files, selected_mask_files)):
            img = np.load(os.path.join(img_dir, img_file))
            mask = np.load(os.path.join(mask_dir, mask_file))

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.subplots_adjust(wspace=0.05)

            # Plot RGB or grayscale image
            if img.ndim == 3 and img.shape[2] >= 3:
                try:
                    rgb = img[:, :, list(rgb_indices)]
                    axes[0].imshow(rgb)
                except IndexError:
                    print(f"Image {img_file} doesn't have enough channels. Showing band 0.")
                    axes[0].imshow(img[:, :, 0], cmap='gray')
            else:
                axes[0].imshow(img.squeeze(), cmap='gray')
            axes[0].set_title('Image (RGB)')
            axes[0].axis('off')

            # Plot mask and optionally overlay values
            axes[1].imshow(mask.squeeze(), cmap='nipy_spectral')
            overlay_class_labels(
                ax=axes[1],
                mask=mask,
                max_per_class=num_pixels_per_class,
                show_values=plot_pixel_values
            )
            axes[1].set_title('Mask (with labels)' if plot_pixel_values else 'Mask')
            axes[1].axis('off')

            out_path = os.path.join(output_dir, f"{prefix}_sample_{i}.png")
            plt.savefig(out_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved: {out_path}")

    print("Plotting training samples...")
    plot_samples(train_img_dir, train_mask_dir, prefix="train")

    print("Plotting validation samples...")
    plot_samples(val_img_dir, val_mask_dir, prefix="val")

def plot_and_save_npy_samples_after_remap(
    train_img_dir, train_mask_dir,
    val_img_dir, val_mask_dir,
    output_dir="sample_plots",
    num_samples=5,
    rgb_indices=(3, 2, 1),
    ignore_label=255,
    plot_pixel_values=True,
    num_pixels_per_class=2
):
    """
    plot the same npy files in 2 folder train and val to check the remap function
    """
    
    os.makedirs(output_dir, exist_ok=True)

    def overlay_class_labels(ax, mask, max_per_class, show_values):
        """Overlay class label values at random pixels on the mask."""
        mask = mask.squeeze()
        unique_classes = np.unique(mask)
        class_counts = {cls: 0 for cls in unique_classes if cls != ignore_label}

        for cls in class_counts.keys():
            positions = np.argwhere(mask == cls)
            if len(positions) == 0:
                continue
            selected = positions[np.random.choice(len(positions), min(max_per_class, len(positions)), replace=False)]
            for y, x in selected:
                if show_values:
                    ax.text(
                        x, y, str(cls),
                        color='white', fontsize=6,
                        ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
                    )

    def plot_sample_pair(idx, fname):
        train_img = np.load(os.path.join(train_img_dir, fname))
        train_mask = np.load(os.path.join(train_mask_dir, fname))
        val_img = np.load(os.path.join(val_img_dir, fname))
        val_mask = np.load(os.path.join(val_mask_dir, fname))

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(wspace=0.05, hspace=0.15)

        # Top: Train image & mask
        if train_img.ndim == 3 and train_img.shape[2] >= 3:
            try:
                rgb_train = train_img[:, :, list(rgb_indices)]
                axes[0, 0].imshow(rgb_train, vmax=30)
            except IndexError:
                axes[0, 0].imshow(train_img[:, :, 0], cmap='gray',vmax=30)
        else:
            axes[0, 0].imshow(train_img.squeeze(), cmap='gray',vmax=30)
        axes[0, 0].set_title('Train Image (RGB)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(train_mask.squeeze(), cmap='nipy_spectral')
        overlay_class_labels(axes[0, 1], train_mask, max_per_class=num_pixels_per_class, show_values=plot_pixel_values)
        axes[0, 1].set_title('Train Mask')
        axes[0, 1].axis('off')

        # Bottom: Val image & mask
        if val_img.ndim == 3 and val_img.shape[2] >= 3:
            try:
                rgb_val = val_img[:, :, list(rgb_indices)]
                axes[1, 0].imshow(rgb_val)
            except IndexError:
                axes[1, 0].imshow(val_img[:, :, 0], cmap='gray')
        else:
            axes[1, 0].imshow(val_img.squeeze(), cmap='gray')
        axes[1, 0].set_title('Val Image (RGB)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(val_mask.squeeze(), cmap='nipy_spectral')
        overlay_class_labels(axes[1, 1], val_mask, max_per_class=num_pixels_per_class, show_values=plot_pixel_values)
        axes[1, 1].set_title('Val Mask')
        axes[1, 1].axis('off')

        out_path = os.path.join(output_dir, f"paired_sample_{idx}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

    # Use filenames common to both train and val dirs
    common_filenames = sorted(
        list(set(os.listdir(train_img_dir)) & set(os.listdir(val_img_dir)))
    )
    common_npy_files = [f for f in common_filenames if f.endswith('.npy')]

    selected_files = random.sample(common_npy_files, min(num_samples, len(common_npy_files)))

    for i, fname in enumerate(selected_files):
        plot_sample_pair(i, fname)

def check_npy_shapes(folder):
    shapes = {}
    for fname in os.listdir(folder):
        if fname.endswith(".npy"):
            arr = np.load(os.path.join(folder, fname))
            shape = arr.shape
            shapes[shape] = shapes.get(shape, 0) + 1

    for shape, count in shapes.items():
        print(f"Shape {shape}: {count} files")
        
def main():
    #CHANGE NAME OF INPUT IMAGE MUST CHANGE THE NAME OF MASK TIFF, 
    #Code mask the same name of image in both folder
    
   
    #--Data augmentation process
    image_dir="/media/tahoangtrung/workSSD/TrainingModelData/InputTiff"
    mask_dir="/media/tahoangtrung/workSSD/TrainingModelData/MaskTiff"   
    patch_image_dir="/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_Patch"  
    patch_mask_dir="/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_MaskPatch"
    filter_patch_image_dir="/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_Patch_filter"
    filter_patch_mask_dir="/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_MaskPatch_filter"
    aug_image_dir="/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_Patch_aug"
    aug_mask_dir="/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_MaskPatch_aug"
    val_img_dir='/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_Patch_aug_val'
    val_mask_dir='/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_MaskPatch_aug_val'
    test_img_dir='/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_Patch_aug_test'
    test_mask_dir='/media/tahoangtrung/workSSD/TrainingModelData/S12_25bands_MaskPatch_aug_test'
    
   
    
    num_classes = 18
    
    IGNORE_LABEL = 255  # pixels with this value will be ignored

    patch_size=(128, 128)
    
    #use in build augment plan function to decide how many percent of patch (count from top should be re-augmentation)
    top_patch_percent=1
    
    # use in compute aug factor, decide at least how many time augmentation is applied
    base_factor = 2.8

    #percentage of moving from training set to validation set and test set
    percentage= 15  
   
    stride=14
    
    #used in compute factor to decide the weighted should be applied on the percetage of classes
    apply_percent=0.3
    
    #---END OF PARAMETERS
        
    # Step 1: Patch image and mask
    # Change start_index if patch the same dataset more than twice, now start from
    start_index_set=1
    create_patches(image_dir=image_dir,mask_dir=mask_dir,patch_image_dir=patch_image_dir,patch_mask_dir=patch_mask_dir,stride=stride,num_classes=num_classes,start_index=start_index_set,patch_size=patch_size)
     
    # Step 2: Filter noise patches
    # Change min_valid_pixel_ratio value to filter more meaningful data, now the percentage is
    valid_ratio = 0.7
    filter_validation_patches(mask_dir=patch_mask_dir,image_dir=patch_image_dir,output_mask_dir=filter_patch_mask_dir,output_image_dir=filter_patch_image_dir,ignore_label=IGNORE_LABEL, min_valid_pixel_ratio=valid_ratio,min_unique_classes=1)
    
    #10% for validation set
    move_matching_files(filter_patch_image_dir,filter_patch_mask_dir, val_img_dir,val_mask_dir, percentage)
    
    #10% for test set
    move_matching_files(filter_patch_image_dir,filter_patch_mask_dir, test_img_dir,test_mask_dir, percentage)

    # Step 3 : Count pixel to calulate the necessary augmentation times
    original_pixels = count_class_pixels(filter_patch_mask_dir,IGNORE_LABEL)
    
    # Step 4: Compute per-class augmentation factors
    aug_factors = compute_aug_factors(original_pixels, base_factor,apply_percent)
    print("⚖️  Augmentation factors per class:", aug_factors)
    
    # Step 5: Build patch-wise augmentation plan
    aug_plan,total_aug_pixel,active_pixel = build_aug_plan(filter_patch_mask_dir, aug_factors, IGNORE_LABEL,top_patch_percent)

    augment_patches_with_plan(filter_patch_image_dir,filter_patch_mask_dir,aug_plan,aug_image_dir,aug_mask_dir,num_classes,IGNORE_LABEL)
    
    # repale zero value (from augmentation process) to 255
    replace_zero_with_255_in_npy(aug_mask_dir)
    
    #Step 8: Check the final result    
    #Visual check
    
    plot_and_save_npy_samples(
    train_img_dir=aug_image_dir,
    train_mask_dir=aug_mask_dir,
    val_img_dir=val_img_dir,
    val_mask_dir=val_mask_dir,
    output_dir="/home/tahoangtrung/Desktop/LULCVIETNAM/Check_AlphaEarth",
    num_samples=30,
    rgb_indices=(16,15,14),  # B4, B3, B2 for RGB composite
    ignore_label=255,
    plot_pixel_values=True,              # Toggle value display
    num_pixels_per_class=5              # Number of pixels to annotate per class
)
    

   
    #Check number of categories
    #scan_npy_mask_labels(aug_mask_dir)
    
    
    #check_npy_shapes(aug_mask_dir)
    
if __name__ == "__main__":
    main()
    print('finish')