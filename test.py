import os
import csv
import logging
import math
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm

from src.hf.modeling_gend import GenD

logger = logging.getLogger(__name__)


def process_flat_frames_to_csv_model(test_dir, output_file="video_results.csv", batch_size=32):
    # 1. Load the pre-trained GenD model from Hugging Face
    logger.info("Loading GenD model from Hugging Face...")
    model = GenD.from_pretrained("yermandy/GenD_CLIP_L_14")
    model.eval()
    logger.info("Model loaded and set to eval mode.")
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    classes = ["real", "deepfake"]
    
    # Dictionary to group frame paths by video
    # Key: (original_class, video_name) -> Value: [list of frame image paths]
    grouped_videos = defaultdict(list)
    
    logger.info("Mapping and grouping frames by video source...")
    
    # 2. Scan 'real' and 'deepfake' folders to group the flat frames
    for current_class in tqdm(classes, desc="Scanning folders", unit="folder"):
        class_path = os.path.join(test_dir, current_class)
        if not os.path.exists(class_path):
            logger.warning("Folder %s not found.", class_path)
            continue

        file_names = os.listdir(class_path)
        logger.info("Scanning %s files in %s...", len(file_names), class_path)

        for file_name in tqdm(file_names, desc=f"Grouping {current_class}", unit="file", leave=False):
            if file_name.lower().endswith(valid_extensions):
                full_path = os.path.join(class_path, file_name)
                
                # Remove the file extension (e.g., .png)
                name_without_ext = os.path.splitext(file_name)[0]
                
                # Split by the last underscore to remove the frame index suffix
                parts = name_without_ext.rsplit('_', 1)
                video_name = parts[0]
                
                # Group using both the folder class and the extracted video prefix
                grouped_videos[(current_class, video_name)].append(full_path)

    total_frames = sum(len(frame_paths) for frame_paths in grouped_videos.values())
    logger.info("Mapping completed. Found %s videos and %s frames.", len(grouped_videos), total_frames)
    final_results = []

    # 3. Process each grouped video
    grouped_items = sorted(grouped_videos.items())
    for (original_class, video_name), frame_paths in tqdm(grouped_items, desc="Processing videos", unit="video"):
        # Sort paths to keep the correct frame sequence order (e.g., _001, _002)
        frame_paths.sort()
        
        logger.info("Processing '%s' (%s) with %s frames...", video_name, original_class, len(frame_paths))
        
        real_accumulator = 0.0
        fake_accumulator = 0.0
        total_processed_frames = 0

        # Process frames in mini-batches to prevent VRAM/RAM overflow
        total_batches = math.ceil(len(frame_paths) / batch_size)
        batch_iterator = range(0, len(frame_paths), batch_size)
        for i in tqdm(
            batch_iterator,
            desc=f"{original_class}/{video_name}",
            total=total_batches,
            unit="batch",
            leave=False,
        ):
            current_batch = frame_paths[i : i + batch_size]
            pil_images = []
            
            for img_path in current_batch:
                img = Image.open(img_path).convert('RGB')
                pil_images.append(img)
            
            # Preprocess and stack images into a batch tensor
            tensors = torch.stack([model.feature_extractor.preprocess(img) for img in pil_images])
            
            with torch.no_grad():
                logits = model(tensors)
                probs = logits.softmax(dim=-1)
            
            for p in probs:
                real_accumulator += p[0].item()
                fake_accumulator += p[1].item()
                total_processed_frames += 1

        # 4. Calculate the average probability for the entire video
        mean_real = real_accumulator / total_processed_frames
        mean_fake = fake_accumulator / total_processed_frames
        logger.info(
            "Finished '%s' (%s): frames=%s, p_real=%.4f, p_fake=%.4f",
            video_name,
            original_class,
            total_processed_frames,
            mean_real,
            mean_fake,
        )

        # Map original folder class to integer label (0: real, 1: deepfake)
        if original_class == "real":
            label = 0
        else:
            label = 1

        # Decision logic mapping to integer predictions (0: real, 1: deepfake)
        if mean_real > mean_fake:
            prediction = 0
        else:
            prediction = 1

        # Reconstruct the expected file path string format
        file_path = f"{original_class}/{video_name}.mp4"

        # Store structured data exactly matching the provided CSV schema
        final_results.append({
            "file_path": file_path,
            "label": label,
            "prediction": prediction,
            "score": mean_fake,  # Score represents the deepfake class probability
            "num_frames": total_processed_frames
        })

    # 5. Export consolidated statistics to a CSV file matching the required schema
    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        header = ["file_path", "label", "prediction", "score", "num_frames"]
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for row in tqdm(final_results, desc="Writing CSV", unit="row"):
            writer.writerow(row)
            
    logger.info("Execution finished. Summary report saved to: %s", output_file)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Target directory path containing 'real' and 'deepfake' subfolders
    TEST_DIR_PATH = "FakeParts/test"
    
    process_flat_frames_to_csv_model(TEST_DIR_PATH)
