#!/usr/bin/env python3
"""
Duplicate video finder using perceptual frame hashing.
Keeps the largest file from each duplicate group, moves the rest to a _duplicates folder.
Usage: python3 find_duplicate_videos.py "/path/to/folder"
"""

import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
from itertools import combinations

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.wmv', '.flv', '.webm'}
FRAMES_TO_SAMPLE = 10
HASH_SIZE = 16
SIMILARITY_THRESHOLD = 90  # % similarity to consider duplicate


def dhash(image, hash_size=HASH_SIZE):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return diff.flatten()


def hash_distance(h1, h2):
    return np.count_nonzero(h1 != h2)


def similarity_percent(h1, h2):
    return (1 - hash_distance(h1, h2) / (HASH_SIZE * HASH_SIZE)) * 100


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
        return False, 0
    similarities = [similarity_percent(h1, h2) for h1, h2 in zip(hashes1, hashes2)]
    avg_similarity = np.mean(similarities)
    return avg_similarity >= SIMILARITY_THRESHOLD, avg_similarity


def find_videos(folder):
    return sorted([f for f in Path(folder).rglob('*') if f.suffix.lower() in VIDEO_EXTENSIONS])


def main():
    if len(sys.argv) < 2:
        folder = input("Enter the path to your video folder: ").strip().strip('"').strip("'")
    else:
        folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid folder.")
        sys.exit(1)

    folder = Path(folder)
    duplicates_folder = folder / "_duplicates"

    print(f"\nScanning: {folder}")
    videos = find_videos(folder)

    if not videos:
        print("No video files found.")
        sys.exit(0)

    print(f"Found {len(videos)} video(s). Analyzing frames...\n")

    video_hashes = {}
    for i, video in enumerate(videos, 1):
        # skip anything already in the _duplicates folder
        if '_duplicates' in video.parts:
            continue
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
        print("No duplicate videos found! Nothing moved.")
    else:
        print(f"Found {len(duplicate_groups)} group(s) of duplicates.\n")
        duplicates_folder.mkdir(exist_ok=True)
        total_moved = 0

        for i, group in enumerate(duplicate_groups, 1):
            # Keep the largest file, move the rest
            sorted_group = sorted(group, key=lambda v: v.stat().st_size, reverse=True)
            keep = sorted_group[0]
            to_move = sorted_group[1:]

            print(f"  Group {i}:")
            print(f"    KEEP:  {keep.name}  ({keep.stat().st_size / (1024*1024):.1f} MB)")
            for v in to_move:
                dest = duplicates_folder / v.name
                # avoid overwriting if same filename already in _duplicates
                if dest.exists():
                    dest = duplicates_folder / (v.stem + "_dup" + v.suffix)
                shutil.move(str(v), str(dest))
                print(f"    MOVED: {v.name}  ({v.stat().st_size / (1024*1024):.1f} MB) -> _duplicates/")
                total_moved += 1
            print()

        print(f"Done! {total_moved} duplicate(s) moved to: {duplicates_folder}")
        print("Nothing was deleted — check the _duplicates folder before removing anything.")

    print(f"\nTotal videos scanned: {len(video_hashes)}")
    print(f"Duplicate groups found: {len(duplicate_groups)}")


if __name__ == "__main__":
    main()
