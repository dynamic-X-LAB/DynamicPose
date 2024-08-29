import os, sys
import traceback
import random

pwd=os.getcwd()
print(pwd)
sys.path.append(pwd)

from PIL import Image
import cv2
from controlnet_aux.util import HWC3
from src.utils.util import get_fps, read_frames, save_videos_from_pil
from tools.utils import read_pts_from_jsonfile_compatible
from src.dwpose import draw_pose_new
from .infer_rtm import draw_pose_0411

def get_files(dir, type='.mp4'):
    video_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(type):
                file_path = os.path.join(root, file)
                #print(file_path)
                video_paths.append(file_path)
    return video_paths

def draw_info_pose(info,version='0330'):
    if version == '0330':
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
        scores = dict(foot_score=info['foot_score'], hands_score=info['hands_score'], body_score=info['bodies.score'])

        detected_map = draw_pose_new(pose, scores, H, W)
        # detected_map = HWC3(detected_map)
        #detected_map = Image.fromarray(detected_map)
        detected_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
    elif version == '0411':
        info_pts = {
        'W': info["W"],
        'H': info["H"],
        'body_points': info['bodies.candidate2'].tolist(),
        'body_scores': info['bodies.score2'].tolist(),
        'hands_left': info['hands'][0].tolist(),
        'hands_left_score': info['hands_score'][0:16].tolist(),
        'hands_right': info['hands'][1].tolist(),
        'hands_right_score': info['hands_score'][16:32].tolist(),
        'feet_left':info['foot'][0:3].tolist(),
        'feet_left_score':info['foot_score'][0:3].tolist(),
        'feet_right':info['foot'][3:6].tolist(),
        'feet_right_score': info['foot_score'][3:6].tolist(),
        }
        detected_map = draw_pose_0411(info_pts)

    return detected_map


    
    
