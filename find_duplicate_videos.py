#!/usr/bin/env python3
"""
Duplicate video finder using perceptual frame hashing.
Usage: python find_duplicate_videos.py "/path/to/folder"
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.flv', '.webm'}
FRAMES_TO_SAMPLE = 10  # frames sampled per video for comparison
HASH_SIZE = 16
SIMILARITY_THRESHOLD = 90  # % similarity to consider duplicate


def dhash(image, hash_size=HASH_SIZE):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return diff.flatten()


def hash_distance(h1, h2):
    return np.count_nonzero(h1 != h2)


def max_hash_bits():
    return HASH_SIZE * HASH_SIZE


def similarity_percent(h1, h2):
    return (1 - hash_distance(h1, h2) / max_hash_bits()) * 100


def extract_frame_hashes(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Could not open: {video_path.name}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    sample_positions = np.linspace(0, total_frames - 1, FRAMES_TO_SAMPLE, dtype=int)
    hashes = []

    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if ret:
            hashes.append(dhash(frame))

    cap.release()
    return hashes if hashes else None


def videos_are_duplicates(hashes1, hashes2):
    if not hashes1 or not hashes2:
        return False
    similarities = []
    for h1, h2 in zip(hashes1, hashes2):
        similarities.append(similarity_percent(h1, h2))
    avg_similarity = np.mean(similarities)
    return avg_similarity >= SIMILARITY_THRESHOLD, avg_similarity


def find_videos(folder):
    folder = Path(folder)
    videos = [f for f in folder.rglob('*') if f.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def main():
    if len(sys.argv) < 2:
        folder = input("Enter the path to your video folder: ").strip().strip('"').strip("'")
    else:
        folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid folder.")
        sys.exit(1)

    print(f"\nScanning: {folder}")
    videos = find_videos(folder)

    if not videos:
        print("No video files found.")
        sys.exit(0)

    print(f"Found {len(videos)} video(s). Analyzing frames...\n")

    # Check for opencv
    try:
        import cv2
    except ImportError:
        print("Missing dependency. Run: pip install opencv-python numpy")
        sys.exit(1)

    video_hashes = {}
    for i, video in enumerate(videos, 1):
        print(f"  [{i}/{len(videos)}] {video.name}")
        hashes = extract_frame_hashes(video)
        if hashes:
            video_hashes[video] = hashes

    print("\nComparing videos for duplicates...\n")

    duplicate_groups = []
    checked = set()

    for v1, v2 in combinations(video_hashes.keys(), 2):
        pair = (v1, v2)
        if pair in checked:
            continue
        checked.add(pair)

        is_dup, similarity = videos_are_duplicates(video_hashes[v1], video_hashes[v2])
        if is_dup:
            placed = False
            for group in duplicate_groups:
                if v1 in group or v2 in group:
                    group.add(v1)
                    group.add(v2)
                    placed = True
                    break
            if not placed:
                duplicate_groups.append({v1, v2})

    if not duplicate_groups:
        print("No duplicate videos found!")
    else:
        print(f"Found {len(duplicate_groups)} group(s) of duplicates:\n")
        for i, group in enumerate(duplicate_groups, 1):
            print(f"  Duplicate Group {i}:")
            for v in sorted(group):
                size_mb = v.stat().st_size / (1024 * 1024)
                print(f"    - {v.name}  ({size_mb:.1f} MB)")
            print()

    print(f"\nTotal videos scanned: {len(video_hashes)}")
    print(f"Duplicate groups found: {len(duplicate_groups)}")


if __name__ == "__main__":
    main()
