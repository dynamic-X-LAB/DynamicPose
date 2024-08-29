import concurrent.futures
import os
import random
from pathlib import Path
import cv2 
import multiprocessing

import numpy as np

import os
import logging
from datetime import datetime

import json

def get_files(dir, type='.mp4'):
    video_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(type):
                file_path = os.path.join(root, file)
                video_paths.append(file_path)
    return video_paths

def get_new_logger(log_dir=None, log_name=None):
    if log_dir is None:
        log_dir = './log_dir'
    if log_name is None:
        log_name = 'log_{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

    if not os.path.exists(log_dir):
        print("[LOG_INFO_CSJ] Buid new log dir:%s"%log_dir)
        os.makedirs(log_dir)
    log_file_path = '%s/%s'%(log_dir, log_name)
    print("[LOG_INFO_CSJ] log_file_path:%s"%log_file_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter( '[LOG_INFO_CSJ] %(asctime)s %(filename)s %(lineno)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
    
def read_pts_from_jsonfile_compatible(json_path):
    info_dict_list = []
    f = open(json_path, 'r')
    lines = f.readlines()
    f.close()
    raw_path = ''
    tags = {}

    version = 'unknown'
    tag_version = 'unknown'
    if 'save_dict_all' in lines[0]:
        assert(len(lines) == 1), len(lines)
        temp = json.loads(lines[0])

        for tag, value in temp.items():
            if tag == 'save_dict_all':
                info_all = value
                continue
            elif tag == 'raw_path':
                raw_path = value
                continue

            if tag in ['cloth_tag', 'backgroud', 'caption', 'background_flow','fps']:
                tags[tag] = float(value)
            elif tag in ['caption_frame_num', 'quality']:
                tags[tag] = value 
            elif tag == 'version':
                version = value
                tags['version'] = value
            elif tag == 'tag_version':
                tags['tag_version'] = value
            else:
                raise ValueError(f"tag unknown:{tag}, {json_path}")
        
    else:
        info_all = []
        for line in lines:
            info = json.loads(line)
            info_all.append(info)

        if len(info_all) == 1 and isinstance(info_all[0], list):
            info_all = info_all[0]
        if 'version' in info_all[0].keys():
            version = info_all[0]['version']

    if version == "unknown":
        print(f"unknown version {json_path}")
        version = '0411'
    for info in info_all:

        out_info = {
            'W': info['W'], 
            'H': info['H'], 
            'index': info['index'], 
            }
        
        for key in ['foot', 'faces', 'hands', "bodies.candidate", "bodies.subset","foot_score","hands_score", "bodies.score"]:
            if key not in info.keys():
                continue
            value_str = info[key]
            if key == "bodies.subset" or key == "foot_score" or key == "hands_score" or key == "bodies.score":
                value = np.fromstring(value_str.replace('[', '').replace(']', ''), sep=" ")
            else:
                value = np.fromstring(value_str.replace('[', '').replace(']', ''), sep=" ").reshape(-1,2)
                if key == "hands":
                    if int(version) >= int('0411'):
                        value = np.array([value[0:16], value[16:32]])
                    else:
                        value = np.array([value[0:21], value[21:42]])

            out_info[key] = value

        index_map = [0,17,6,8,10,5,7,9,12,14,16,11,13,15,2,1,4,3]
        out_info['bodies.candidate2'] = np.take(out_info['bodies.candidate'], index_map, axis=0)
        out_info['bodies.score2'] = np.take(out_info['bodies.candidate'], index_map, axis=0)

        out_info['bodies.candidate'] = np.take(out_info['bodies.candidate'], index_map, axis=0) / np.array([info['W'], info['H']])

        if 'bodies.score' in out_info.keys():
            out_info['bodies.score'] = np.take(out_info['bodies.score'], index_map, axis=0)
        else:
            out_info['bodies.score'] = np.ones([len(index_map)], dtype=np.float32)

        info_dict_list.append(out_info)

    res = {
            "info_dict_list": info_dict_list,
            "raw_path": raw_path,
            }
    res.update(tags)

    return res
    
