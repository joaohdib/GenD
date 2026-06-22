import os
import csv
import logging
import math
import argparse
import traceback
from collections import defaultdict

import torch
from PIL import Image
from PIL import UnidentifiedImageError
from tqdm import tqdm

from src.hf.modeling_gend import GenD

logger = logging.getLogger(__name__)


CSV_HEADER = ["file_path", "label", "prediction", "score", "num_frames"]
ERROR_HEADER = ["file_path", "error_type", "error_message", "traceback"]


def _read_completed_file_paths(output_file):
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return set()

    completed = set()
    try:
        with open(output_file, mode="r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                file_path = row.get("file_path")
                if file_path:
                    completed.add(file_path)
    except (OSError, csv.Error) as exc:
        logger.warning(
            "Could not read existing CSV %s for resume support: %s",
            output_file,
            exc,
        )

    return completed


def _open_csv_for_append(path, header):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    csv_file = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=header)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()
        os.fsync(csv_file.fileno())
    return csv_file, writer


def _write_row_and_sync(csv_file, writer, row):
    writer.writerow(row)
    csv_file.flush()
    os.fsync(csv_file.fileno())


def _safe_load_image(img_path):
    try:
        with Image.open(img_path) as img:
            return img.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        logger.warning("Skipping unreadable frame %s: %s", img_path, exc)
    return None


def process_flat_frames_to_csv_model(test_dir, output_file="video_results.csv", batch_size=32):
    if batch_size < 1:
        logger.warning("Invalid batch_size=%s. Falling back to batch_size=1.", batch_size)
        batch_size = 1

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

    completed_file_paths = _read_completed_file_paths(output_file)
    if completed_file_paths:
        logger.info("Resume enabled: found %s completed videos in %s.", len(completed_file_paths), output_file)

    error_file = f"{os.path.splitext(output_file)[0]}_errors.csv"

    # 3. Process each grouped video
    grouped_items = sorted(grouped_videos.items())
    output_csv, output_writer = _open_csv_for_append(output_file, CSV_HEADER)
    error_csv, error_writer = _open_csv_for_append(error_file, ERROR_HEADER)
    try:
        for (original_class, video_name), frame_paths in tqdm(grouped_items, desc="Processing videos", unit="video"):
            file_path = f"{original_class}/{video_name}.mp4"
            if file_path in completed_file_paths:
                logger.info("Skipping already completed video: %s", file_path)
                continue

            try:
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
                    image_items = []

                    for img_path in current_batch:
                        img = _safe_load_image(img_path)
                        if img is not None:
                            pil_images.append(img)
                            image_items.append((img_path, img))

                    if not pil_images:
                        continue

                    try:
                        # Preprocess and stack images into a batch tensor
                        tensors = torch.stack([model.feature_extractor.preprocess(img) for img in pil_images])

                        with torch.no_grad():
                            logits = model(tensors)
                            probs = logits.softmax(dim=-1)

                        for p in probs:
                            real_accumulator += p[0].item()
                            fake_accumulator += p[1].item()
                            total_processed_frames += 1
                    except Exception as exc:
                        logger.exception(
                            "Batch failed for %s at frame offset %s. Retrying frames individually: %s",
                            file_path,
                            i,
                            exc,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        for img_path, img in image_items:
                            try:
                                tensor = model.feature_extractor.preprocess(img).unsqueeze(0)
                                with torch.no_grad():
                                    logits = model(tensor)
                                    probs = logits.softmax(dim=-1)

                                p = probs[0]
                                real_accumulator += p[0].item()
                                fake_accumulator += p[1].item()
                                total_processed_frames += 1
                            except Exception as frame_exc:
                                logger.exception(
                                    "Skipping frame after individual retry failed: %s (%s)",
                                    img_path,
                                    frame_exc,
                                )
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                if total_processed_frames == 0:
                    raise RuntimeError("No readable frames were processed for this video.")

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

                row = {
                    "file_path": file_path,
                    "label": label,
                    "prediction": prediction,
                    "score": mean_fake,  # Score represents the deepfake class probability
                    "num_frames": total_processed_frames,
                }
                _write_row_and_sync(output_csv, output_writer, row)
                completed_file_paths.add(file_path)
            except Exception as exc:
                logger.exception("Failed to process video %s. Continuing with next video.", file_path)
                _write_row_and_sync(
                    error_csv,
                    error_writer,
                    {
                        "file_path": file_path,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        output_csv.close()
        error_csv.close()
            
    logger.info("Execution finished. Summary report saved to: %s", output_file)
    logger.info("Errors, if any, were saved to: %s", error_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process test frames from real/deepfake folders and export video predictions to CSV."
    )
    parser.add_argument(
        "test_dir_path",
        help="Path to the test directory containing 'real' and 'deepfake' subfolders.",
    )
    parser.add_argument(
        "--output-file",
        default="video_results.csv",
        help="CSV output file path. Default: video_results.csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of frames processed per batch. Default: 32",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    process_flat_frames_to_csv_model(
        args.test_dir_path,
        output_file=args.output_file,
        batch_size=args.batch_size,
    )
