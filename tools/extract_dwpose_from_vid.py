import concurrent.futures
import os
import random
from pathlib import Path
import cv2 
import multiprocessing

import numpy as np
import sys
pwd=os.getcwd()
print(pwd)
sys.path.append(pwd)

from src.dwpose import DWposeDetector 
from data_prepare import draw_pts
from src.utils.util import get_fps, read_frames, save_videos_from_pil

import time
import json
from PIL import Image
import glob

def read_images(image_dir):
    image_paths = set()
    for root, dirs, files in os.walk(image_dir):
        for name in files:
            if name.endswith(".png") or name.endswith(".jpg"):
                image_paths.add(os.path.join(root, name))
    image_paths = list(image_paths)
    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(image)
    return images, image_paths

def process_single_video(video_path, detector, root_dir, save_dir, image_flag = False):
    start_time = time.time()
    print(f"video_path={video_path}")
    
    if not image_flag:
        relative_path = os.path.relpath(video_path, root_dir)
        print(relative_path, video_path, root_dir)
        out_path = os.path.join(save_dir, relative_path)
        if os.path.exists(out_path):
            print(f"Warning: out_path={out_path} exists, so not process again!!!!")
            return

        output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path) if not image_flag else 0
    if not image_flag:
        frames = read_frames(video_path)
    else:
        frames, image_names = read_images(video_path)
    kps_results = []
    save_dict_all = []
    pose_time_all = 0
    draw_time_all = 0
    for i, frame_pil in enumerate(frames):
        pose_time, draw_time= 0, 0
        result, score, save_dict = detector(frame_pil)
        score = np.mean(score, axis=-1)
        save_dict_all.append(save_dict)
        kps_results.append(result)
        pose_time_all += pose_time
        draw_time_all += draw_time


    end_time2 = time.time()
    
    if not image_flag:
        save_videos_from_pil(kps_results, out_path, fps=fps)

        # save kps info
        print(f"out_path={out_path}")
        f = open(f'{out_path}.json', 'w')
        for i, info in enumerate(save_dict_all): 
            info['index'] = i 
            save_dict_info = json.dumps(info)
            f.write(save_dict_info+'\n')
        f.close()
    else:
        image_num = len(image_names)
        for i in range(image_num):
            info = save_dict_all[i]
            info['index'] = 0 
            image_name = image_names[i]
            f = open(f'{image_name}.json', 'w')
            save_dict_info = json.dumps(info)
            f.write(save_dict_info+'\n')
            f.close()


    end_time = time.time()

    print(f"pose image time (s): {pose_time_all}, f={len(frames)}")
    print(f"draw image time (s): {draw_time_all}")
    print(f"save video time (s): {end_time - end_time2}")
    print(f"all time (s): {end_time - start_time}")


def process_batch_videos(video_list, detector, root_dir, save_dir):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, root_dir, save_dir)

def read_pts_from_jsonfile(json_path):
    assert("Deprecated, using read_pts_from_jsonfile_compatible please")
    info_dict_list = []
    f = open(json_path, 'r')
    lines = f.readlines()

    for line in lines:
        info = json.loads(line)
        #print(info)
        out_info = {
            'W': info['W'], 
            'H': info['H'], 
            'index': info['index'], 
            }
        for key in ['foot', 'faces', 'hands', "bodies.candidate", "bodies.subset", "bodies.score"]:
            if key not in info.keys():
                continue
            value_str = info[key]
            if key == "bodies.subset" or key == "bodies.score":
                value = np.fromstring(value_str.replace('[', '').replace(']', ''), sep=" ")
            else:
                value = np.fromstring(value_str.replace('[', '').replace(']', ''), sep=" ").reshape(-1,2)
                if key == "hands":
                    value = np.array([value[0:21], value[21:42]])
            out_info[key] = value

        info_dict_list.append(out_info)
    return info_dict_list
    
def read_pts_from_jsonfile2(json_path):
    info_dict_list = []
    f = open(json_path, 'r')
    lines = f.readlines()
    assert(len(lines) == 1), len(lines)
    info_all = json.loads(lines[0])
    info_all = info_all['save_dict_all']
    for info in info_all:
        #print(info)
        out_info = {
            'W': info['W'], 
            'H': info['H'], 
            'index': info['index'], 
            }
        for key in ['foot', 'faces', 'hands', "bodies.candidate", "bodies.subset"]:
            value_str = info[key]
            if key == "bodies.subset":
                value = np.fromstring(value_str.replace('[', '').replace(']', ''), sep=" ")
            else:
                value = np.fromstring(value_str.replace('[', '').replace(']', ''), sep=" ").reshape(-1,2)
                if key == "hands":
                    value = np.array([value[0:21], value[21:42]])
            out_info[key] = value

        info_dict_list.append(out_info)
    return info_dict_list
    
def read_pts_from_jsonfile_compatible(json_path):
    assert("Deprecated, using tools.utils.read_pts_from_jsonfile_compatible please")
    
# get or read pts of one video or in image dir
# for test read kpt json file

from controlnet_aux.util import HWC3
import queue
def draw_pts_map(q, th_id):
    while True:
        try:
            if q.qsize() % 1 == 0:
                print(q.qsize())
            json_path = q.get(timeout=1)
            video_path = json_path.replace("/kpt_", "/video_")[:-5]
            fps = get_fps(video_path)

            json_info = read_pts_from_jsonfile_compatible(json_path)
            info_dict_list = json_info['info_dict_list']
            raw_path = json_info['raw_path']
            cloth_tag = json_info['cloth_tag']

            out_path =  json_path[:-5]

            kps_results = []
            for info in info_dict_list:
                bodies ={}
                W = info["W"]
                H = info["H"]
                bodies["candidate"] = info["bodies.candidate"]
                bodies["subset"] = [info["bodies.subset"]]
                pose = {}
                pose["bodies"] = bodies

                pose["faces"] = info["faces"]
                pose["hands"] = info["hands"]
                pose["foot"] = [info["foot"]]

                #detected_map = draw_pose(pose, H, W)
                detected_map = draw_pts.draw_info_pose(info) #0330
                detected_map = HWC3(detected_map)
                detected_map = Image.fromarray(detected_map)
                kps_results.append(detected_map)

            save_videos_from_pil(kps_results, out_path, fps=fps)
        except queue.Empty:
            print(f"th_id={th_id} queue empty")
            break
        else:
            time.sleep(1)

