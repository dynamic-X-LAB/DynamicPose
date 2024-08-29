import argparse
import json
import os
import time

# -----
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
# -----
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--meta_info_name", type=str)

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name


basename = os.path.basename(args.root_path)
assert('video' in basename), args.root_path
pose_dir = os.path.join( os.path.dirname(args.root_path), basename.replace("video", "kpt"))

# collect all video_folder paths
video_mp4_paths = set()
for root, dirs, files in os.walk(args.root_path):
    for name in files:
        if name.endswith(".mp4"):
            file_path = os.path.join(root, name)
            if 0:
                time_stamp_start = time.mktime(time.strptime("2024-03-04 13:00:00", '%Y-%m-%d %H:%M:%S'))
                time_stamp = os.path.getmtime(file_path)
                if time_stamp < time_stamp_start:
                    continue

            video_mp4_paths.add(file_path)
video_mp4_paths = list(video_mp4_paths)

meta_infos = []
for video_mp4_path in video_mp4_paths:
    relative_video_name = os.path.relpath(video_mp4_path, args.root_path)
    kps_path = os.path.join(pose_dir, relative_video_name)
    meta_infos.append({"video_path": video_mp4_path, "kps_path": kps_path})

json.dump(meta_infos, open(f"./data/meta/{args.meta_info_name}_meta.json", "w"))
