
import os
import numpy as np
import json
import math
from PIL import Image
import cv2
from mmpose.apis import MMPoseInferencer
from moviepy.editor import VideoFileClip
import sys
pwd=os.getcwd()
sys.path.append(pwd)

from src.dwpose import DWposeDetector
from tqdm import tqdm

base_path = os.path.split(os.path.abspath(__file__))[0]

from pathlib import Path

def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")

def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def draw_body(canvas, points, scores, kpt_thr=0.4,stickwidth = 4,r=4):
    colors = [
        [255, 0, 0], # 0
        [0, 255, 0], # 1
        [0, 0, 255], # 2
        [255, 0, 255], # 3
        [255, 255, 0], # 4
        [85, 255, 0], #5
        [0, 75, 255], #6
        [0, 255, 85], #7
        [0, 255, 170], #8
        [170, 0, 255], #9
        [85, 0, 255], #10
        [0, 85, 255], #11
        [0, 255, 255], #12
        [85, 0, 255], #13
        [170, 0, 255], #14
        [255, 0, 255], #15
        [255, 0, 170], #16
        [255, 0, 85], #17
    ]
    connetions = [
        [17,0],[0, 1],[0, 2],[2, 4],[1, 3],
        [17,6],[6,8],[8,10],
        [17,5],[5,7],[7,9],
        [17,12],[12,14],[14,16],
        [17,11],[11,13],[13,15],
    ]
    connection_colors = [
        [255, 0, 0], # 0
        [0, 255, 0], #1
        [0, 0, 255], #2
        [255, 255, 0], #3
        [255, 0, 255], #4
        [0, 255, 0], #5
        [0, 85, 255], #6
        [255, 175, 0], # 7
        [0, 0, 255], ## 8
        [255, 85, 0], #9
        [0, 255, 85], #10
        [255, 0, 255], #11
        [255, 0, 0], #12
        [0, 175, 255], #13
        [255, 255, 0], #14
        [0, 0, 255], #15
        [0, 255, 0], #16
    ]
    if len(points)!=18:
        #  add neck point
        points.append([(points[5][0]+points[6][0])/2,(points[5][1]+points[6][1])/2])
        scores.append((scores[5]+scores[6])/2)

    # draw point
    for i in range(len(points)):
        score = scores[i]
        if i == 15 or i == 16:
            if score < 0.6:
                continue          
        if score < kpt_thr:
            continue
        x,y = points[i][0:2]
        x,y = int(x),int(y)
        cv2.circle(canvas, (x, y), r, colors[i], thickness=-1)
        # cv2.putText(canvas, i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)
    
    # draw line
    for i in range(len(connetions)):
        i = 16 - i
        point1_idx,point2_idx = connetions[i][0:2]

        if point2_idx == 15 or point2_idx == 16:
            if scores[point2_idx] < 0.6:
                continue

        if scores[point1_idx] < kpt_thr or scores[point2_idx] < kpt_thr:
            continue

        point1 = points[point1_idx]
        point2 = points[point2_idx]
        Y = [point2[0],point1[0]]
        X = [point2[1],point1[1]]
        mX = int(np.mean(X))
        mY = int(np.mean(Y))
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((mY, mX), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, connection_colors[i])
        # cv2.putText(canvas, i)+connection_colors[i]), (mY, mX), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)

    return canvas

def draw_foot(canvas, foot, scores, color=[255, 0, 255], kpt_thr=0.4, stickwidth=1,r=1):

    if any(score < kpt_thr for score in scores):
        return canvas
    x, y = foot[2][0:2]
    x1, y1 = foot[0][0:2]
    x2, y2 = foot[1][0:2]
    x, y = int(x), int(y)
    mx = int((x1+x2)/2)
    my = int((y1+y2)/2)

    cv2.circle(canvas, (x, y), r, [255, 255, 255], thickness=-1)
    cv2.circle(canvas, (mx, my), r, [255, 255, 255], thickness=-1)
    cv2.line(canvas, (x, y), (mx, my), color, stickwidth)
    return canvas

def draw_hand(canvas, hand, scores, kpt_thr=0.4, stickwidth=1,r=1):
    # 16 点
    color_finger = [
        [255, 0, 0],    # 大拇指颜色
        [0, 255, 0],    # 食指颜色
        [255, 0, 255],    # 中指颜色
        [0, 255, 255],  # 无名指颜色
        [255, 255, 0],  # 小拇指颜色
    ]

    # 绘制手部关键点
    for i in range(len(hand)):
        score = scores[i]
        if score < kpt_thr:
            continue
        x, y = hand[i][0:2]
        x, y = int(x), int(y)
        cv2.circle(canvas, (x, y), r, [255, 255, 255], thickness=-1)
        # cv2.putText(canvas, i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)

    # 绘制连接线
    for i in range(5): 
        if scores[1 + i*3] >= kpt_thr and scores[0] >= kpt_thr: 
            cv2.line(canvas, tuple(map(int, hand[0][0:2])), tuple(map(int, hand[1 + i*3][0:2])), color_finger[i], stickwidth)
    
        for j in range(1, 3): 
            if scores[1 + i*3 + j - 1] >= kpt_thr and scores[1 + i*3 + j] >= kpt_thr:
                start_point = hand[1 + i*3 + j - 1][0:2]
                end_point = hand[1 + i*3 + j][0:2]
                cv2.line(canvas, tuple(map(int, start_point)), tuple(map(int, end_point)), color_finger[i], stickwidth)

    return canvas

def get_info(body,hand,foot,W,H):
    body_points = body['keypoints']
    body_scores = body['keypoint_scores']

    points = hand['keypoints']
    scores = hand['keypoint_scores']
    hands_idx = [
        92,         # 手根
        94, 95, 96,  # 大拇指
        97, 98, 100,  # 食指
        101, 102, 104,  # 中指
        105, 106, 108,  # 无名指
        109, 110, 112  # 小拇指
    ]
    hands_left = [points[i-1] for i in hands_idx]
    hands_left_score = [scores[i-1] for i in hands_idx]
    hands_right = [points[i+20] for i in hands_idx]
    hands_right_score = [scores[i+20] for i in hands_idx]

    points = foot['keypoints']
    scores = foot['keypoint_scores']

    feet_idx = [18, 19, 20]  
    feet_left = [points[i-1] for i in feet_idx]
    feet_left_score = [scores[i-1] for i in feet_idx]

    feet_right = [points[i+2] for i in feet_idx]
    feet_right_score = [scores[i+2] for i in feet_idx]

    info = {
        'W': W,
        'H': H,
        'body_points': body_points,
        'body_scores': body_scores,
        'hands_left':hands_left,
        'hands_left_score': hands_left_score,
        'hands_right': hands_right,
        'hands_right_score': hands_right_score,
        'feet_left':feet_left,
        'feet_left_score':feet_left_score,
        'feet_right':feet_right,
        'feet_right_score': feet_right_score
    }

    return info

def check_hand(hand):

    finger_indices = [
        [0, 1, 2, 3],  # 大拇指
        [0, 4, 5, 6],  # 食指
        [0, 7, 8, 9],  # 中指
        [0, 10, 11, 12], # 无名指
        [0, 13, 14, 15]  # 小拇指
    ]
    finger_lengths = []

    for finger in finger_indices:
        finger_length = 0
        for i in range(len(finger) - 1):
            point_a = np.array(hand[finger[i]][0:2])
            point_b = np.array(hand[finger[i+1]][0:2])
            distance = np.linalg.norm(point_a - point_b)
            finger_length += distance
        finger_lengths.append(finger_length)

    finger_thresholds = [1, 0.65, 0.6, 0.65, 1]

    for i, length in enumerate(finger_lengths):
        other_lengths = finger_lengths[:i] + finger_lengths[i+1:]
        mean_length = np.mean(other_lengths) 
        threshold_ratio = finger_thresholds[i] 
        
        if length > mean_length * (1 + threshold_ratio) or length < mean_length * (1 - threshold_ratio):
            return False
    return True

def draw_pose_0411(info,frame_pil=None):
    H,W = info['H'],info['W']
    if frame_pil is  None:
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    else:
        canvas = cv2.cvtColor(np.asarray(frame_pil), cv2.COLOR_RGB2BGR)

    # 1. body
    points = info['body_points']
    scores = info['body_scores']
    canvas = draw_body(canvas,points,scores,kpt_thr=0.4,stickwidth = math.ceil(H/256),r=math.ceil(H/256))

    # 2. hands
    hands_left = info['hands_left']
    hands_left_score = info['hands_left_score']
    hands_right = info['hands_right']
    hands_right_score = info['hands_right_score']
    
    if scores[9] > 0.6 and check_hand(hands_left):
        canvas = draw_hand(canvas,hands_left,hands_left_score,kpt_thr=0.65,stickwidth = math.ceil(H/512),r= math.ceil(H/512))
    if scores[10] > 0.6 and check_hand(hands_right):
        canvas = draw_hand(canvas,hands_right,hands_right_score,kpt_thr=0.65,stickwidth = math.ceil(H/512),r= math.ceil(H/512))


    # 3. feet
    feet_left = info['feet_left']
    feet_left_score = info['feet_left_score']
    feet_right = info['feet_right']
    feet_right_score = info['feet_right_score']

    if scores[15] > 0.6:
        canvas = draw_foot(canvas,feet_left,feet_left_score,color=[255,0,255],kpt_thr=0.6,stickwidth = math.ceil(H/256),r=math.ceil(H/256))
    if scores[16] > 0.6:
        canvas = draw_foot(canvas,feet_right,feet_right_score,color=[0,255,0],kpt_thr=0.6,stickwidth = math.ceil(H/256),r=math.ceil(H/256))

    detected_map_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return detected_map_pil

def get_json(info):
    
    save_info = {
        'W': info['W'],
        'H': info['H'],
        'bodies.candidate': str(np.around(info['body_points'],4)),
        'bodies.score':str(np.around(info['body_scores'],2)),
        'hands': str(np.around(info['hands_left']+info['hands_right'],4)),
        'hands_score':str(np.around(info['hands_left_score']+info['hands_right_score'],2)),
        'foot': str(np.around(info['feet_left']+info['feet_right'],4)),
        'foot_score':str(np.around(info['feet_left_score']+info['feet_right_score'],2)),
    }
    return save_info

def get_info2(body,hand,foot,W,H):
    body_points = body['keypoints']
    body_scores = body['keypoint_scores']

    points = hand['keypoints']
    scores = hand['keypoint_scores']
    hands_idx = [
        92,         # 手根
        94, 95, 96,  # 大拇指
        97, 98, 100,  # 食指
        101, 102, 104,  # 中指
        105, 106, 108,  # 无名指
        109, 110, 112  # 小拇指
    ]
    hands_left = [points[i-1] for i in hands_idx]
    hands_left_score = [scores[i-1] for i in hands_idx]
    hands_right = [points[i+20] for i in hands_idx]
    hands_right_score = [scores[i+20] for i in hands_idx]

    points = foot['keypoints']
    scores = foot['keypoint_scores']

    feet_idx = [1, 2, 3]  
    feet_left = [points[i-1] for i in feet_idx]
    feet_left_score = [scores[i-1] for i in feet_idx]

    feet_right = [points[i+2] for i in feet_idx]
    feet_right_score = [scores[i+2] for i in feet_idx]

    info = {
        'W': W,
        'H': H,
        'body_points': body_points,
        'body_scores': body_scores,
        'hands_left':hands_left,
        'hands_left_score': hands_left_score,
        'hands_right': hands_right,
        'hands_right_score': hands_right_score,
        'feet_left':feet_left,
        'feet_left_score':feet_left_score,
        'feet_right':feet_right,
        'feet_right_score': feet_right_score
    }

    return info

def get_files(dir, type='.png'):
    video_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(type):
                file_path = os.path.join(root, file)
                video_paths.append(file_path)
    return video_paths

def process_image(input_dir, output_dir, model_list):
    inferencer_body = model_list[0]
    inferencer_hand = model_list[1]

    origin_path = os.getcwd()
    inferencer_foot = DWposeDetector(foot = True).to("cuda")

    paths = get_files(input_dir, '.png') + get_files(input_dir, '.jpg')
    for path in paths:
        if 1:
            print(f'process {path}')
            image_pil = Image.open(path).convert('RGB')
            frame = np.array(image_pil)
            H,W = frame.shape[0:2]
            #print(path, W, H)

            body = inferencer_body(frame)
            body = next(body)['predictions'][0][0]
            body_w = inferencer_hand(frame)
            hand = next(body_w)['predictions'][0][0]
            feet, feet_scores = inferencer_foot(frame)
            feet = feet.tolist()[0]
            for points in feet:
                points[0] = points[0] * W
                points[1] = points[1] * H
            foot = {
                'keypoints':feet,
                'keypoint_scores':feet_scores.tolist()[0],
            }
            info = get_info2(body,hand,foot, W, H)
            detected_map_pil = draw_pose_0411(info)
            
            pose_map_path = os.path.join(output_dir, os.path.basename(path))
            save_dict = get_json(info)
            save_dict['index'] = 0
            save_dict['version'] = '0411'
            json_frame = pose_map_path+'.json'
            with open(json_frame, 'w') as f:
                f.write(json.dumps(save_dict, ensure_ascii=False))
            if input_dir != output_dir:
                detected_map_pil.save(pose_map_path)

        

def process_video(video_dir, out_dir, model_dict):

    #origin_path = os.getcwd()
    #os.chdir(os.path.join(base_path, "serving"))
    inferencer_foot = DWposeDetector(foot = True).to("cuda")
    #os.chdir(origin_path)
    inferencer_body, inferencer_hand = model_dict


    paths = get_files(video_dir, '.mp4')
    out_path = out_dir

    for path in tqdm(paths):

        print('process: ', path)

        video = VideoFileClip(path)
        fps = video.fps
        first_frame = video.get_frame(0)
        img = np.asarray(first_frame)
        H,W = img.shape[0],img.shape[1]

        pose_frame = []
        save_dict_all = []
        total_frames = int(fps * video.duration)

        for i, frame_pil in tqdm(enumerate(video.iter_frames()), total=total_frames):
            if frame_pil is None:
                break
            frame_pilr = Image.fromarray(frame_pil)
            frame_pilr.save(path+f'_{i}.png')
            frame = np.array(frame_pil)

            body = inferencer_body(frame)
            body = next(body)['predictions'][0][0]
            body_w = inferencer_hand(frame)
            hand = next(body_w)['predictions'][0][0]
            feet, feet_scores = inferencer_foot(frame)
            feet = feet.tolist()[0]
            for points in feet:
                points[0] = points[0] * W
                points[1] = points[1] * H
            foot = {
                'keypoints':feet,
                'keypoint_scores':feet_scores.tolist()[0],
            }
            info = get_info2(body,hand,foot, W, H)
            detected_map_pil = draw_pose_0411(info)
            video_frame = os.path.join(out_path,(path+f'_{i}.png').split("/")[-1])
            detected_map_pil.save(video_frame)
            save_dict = get_json(info)
            save_dict['index'] = i
            json_frame = video_frame+'.json'
            with open(json_frame, 'w') as f:
                f.write(json.dumps(save_dict, ensure_ascii=False))
            save_dict_all.append(save_dict)
            pose_frame.append(detected_map_pil)

        kps_path_new = os.path.join(out_path,(path).split("/")[-1])
        ensure_directory_exists(kps_path_new)
        save_videos_from_pil(pose_frame, kps_path_new, fps=fps)

        json_new = kps_path_new +'.json'
        with open(json_new, 'w') as f:
            f.write(json.dumps(save_dict_all, ensure_ascii=False))

def init_model(model_dir):
    inferencer_body = MMPoseInferencer(
        pose2d = model_dir + '/pretrained_weights/rtmpose/rtmpose-x_8xb256-700e_coco-384x288.py',
        pose2d_weights= model_dir + '/pretrained_weights/rtmpose/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.pth',
        det_model= model_dir + '/pretrained_weights/rtmpose/rtmdet_m_640-8xb32_coco-person.py',
        det_weights = model_dir + '/pretrained_weights/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        det_cat_ids=[0],
    )

    inferencer_hand = MMPoseInferencer(
        pose2d = model_dir + '/pretrained_weights/rtmpose/rtmw-x_8xb320-270e_cocktail14-384x288.py',
        pose2d_weights = model_dir + '/pretrained_weights/rtmpose/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
        det_model = model_dir + '/pretrained_weights/rtmpose/rtmdet_m_640-8xb32_coco-person.py',
        det_weights = model_dir + '/pretrained_weights/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        det_cat_ids=[0],
    )
    return [inferencer_body, inferencer_hand]

if __name__ == '__main__':
    model_dir = './'
    model_list = init_model(model_dir)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    type = sys.argv[3] #'video' or 'image'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    print(f"test {type}")
    if type == 'video':
        process_video(input_dir, output_dir, model_list)
    elif type == 'image':
        process_image(input_dir, output_dir, model_list)


