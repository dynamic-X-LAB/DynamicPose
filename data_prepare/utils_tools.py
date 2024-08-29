import onnxruntime
import datetime
import time
import numpy as np
import cv2
import math
import os
from PIL import Image

import typing as T


def to_seconds(time_str):
    t = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    seconds = 60 * t.minute + 3600 * t.hour + t.second + t.microsecond / 1000000
    return seconds

def resize_image(input_image, resolution):
    H, W = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

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

def get_info3(body,hand,foot,W,H):
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

    face_idx = range(68)
    face =  [points[i+23] for i in face_idx]
    face_score = [scores[i+23] for i in face_idx]

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
        'feet_right_score': feet_right_score,
        'face': face,
        'face_score': face_score
    }

    return info

def check_body(body):
    body_scores = body['keypoint_scores']
    mis_num = 0
    not_check_point = [9,10,15,16,13,14]
    for i,score in enumerate(body_scores):
        if i not in not_check_point:
            if score < 0.3:
                mis_num += 1
    if mis_num > 2:
        return True

    return False

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
    canvas = draw_body(canvas,points,scores,kpt_thr=0.4,stickwidth = H//256,r=H//256)

    # 2. hands
    hands_left = info['hands_left']
    hands_left_score = info['hands_left_score']
    hands_right = info['hands_right']
    hands_right_score = info['hands_right_score']
    
    if scores[9] > 0.6 and check_hand(hands_left):
        canvas = draw_hand(canvas,hands_left,hands_left_score,kpt_thr=0.65,stickwidth = H//512,r= H//512)
    if scores[10] > 0.6 and check_hand(hands_right):
        canvas = draw_hand(canvas,hands_right,hands_right_score,kpt_thr=0.65,stickwidth = H//512,r= H//512)


    # 3. feet
    feet_left = info['feet_left']
    feet_left_score = info['feet_left_score']
    feet_right = info['feet_right']
    feet_right_score = info['feet_right_score']

    if scores[15] > 0.6:
        canvas = draw_foot(canvas,feet_left,feet_left_score,color=[255,0,255],kpt_thr=0.4,stickwidth = H//256,r=H//256)
    if scores[16] > 0.6:
        canvas = draw_foot(canvas,feet_right,feet_right_score,color=[0,255,0],kpt_thr=0.4,stickwidth = H//256,r=H//256)

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

def get_json2(info):
    
    save_info = {
        'W': info['W'],
        'H': info['H'],
        'bodies.candidate': str(np.around(info['body_points'],4)),
        'bodies.score':str(np.around(info['body_scores'],2)),
        'hands': str(np.around(info['hands_left']+info['hands_right'],4)),
        'hands_score':str(np.around(info['hands_left_score']+info['hands_right_score'],2)),
        'foot': str(np.around(info['feet_left']+info['feet_right'],4)),
        'foot_score':str(np.around(info['feet_left_score']+info['feet_right_score'],2)),
        'face': info['face'],
        'face_score': info['face_score'],
    }
    return save_info



def calculate_flow_with_mask(prev_img, curr_img, mask1, mask2):
    lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=50,
                      qualityLevel=0.07,
                      minDistance=7,
                      blockSize=7)

    mask = mask1+mask2
    binary_mask = np.uint8((mask == 0) * 255) # 获取背景mask
    prev_gray = resize_image(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY),resolution=128)
    curr_gray =  resize_image(cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY),resolution=128)

    binary_mask = resize_image(binary_mask,resolution=128)
    kernel = np.ones((10, 10),np.uint8)
    binary_mask = cv2.erode(binary_mask,kernel,iterations = 1)

    height, width = prev_gray.shape

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=binary_mask, **feature_params) 

    if p0 is not None:
        p_num = len(p0.reshape(-1, 2))
        if p_num < 2:
            return 0.0, f'corner_points_not_enough_{p_num}'

        p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        p0r, status, err = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None)


        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
    
        sum_dis = 0
        count_good = 0
        for p00, p11, good_flag in zip(p0.reshape(-1, 2), p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            else:
                sum_dis += np.sqrt(np.sum(np.square(p00 - p11))) 
                count_good += 1
        if count_good < 2:
            return 9.0, f'good_points_not_enough_{count_good}'
        else:
            avg_dis = sum_dis/count_good
            return avg_dis, 'succeed'
    else:
        return 0.0, f'corner_points_not_enough_none'

def check_text_on_person(mask, ocr_result):
    binary_mask = (mask > 0).astype(np.uint8)
    if ocr_result == None:
        return False, 0.
    for item in ocr_result:
        box = item[0],
        words = item[1]
        word, score = words
        if score<0.75:
            continue

        box_array = np.array(box).squeeze(0)
        x_min, y_min = np.min(box_array, axis=0)
        x_max, y_max = np.max(box_array, axis=0)

        percent = np.sum(binary_mask[int(y_min):int(y_max), int(x_min):int(x_max)]) / np.sum(binary_mask)
        if percent > 0.01:
            # print("find: ",word,' percent:',percent)
            return True, percent
    return False, 0.

def get_masked_img(mask, image_array):
    mask_array = (mask > 0).astype(np.uint8)
    result_array = np.zeros_like(image_array)
    result_array = image_array * mask_array[:, :, np.newaxis]
    
    # 设置背景为白色
    background = np.ones_like(image_array) * 255
    background_mask = np.ones_like(mask_array) - mask_array
    result_array += background * background_mask[:, :, np.newaxis]
    
    # result_array = Image.fromarray(result_array.astype(np.uint8))
    return result_array

def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_cloth(img_array, body):
    body_points = body['keypoints']
    keypoints = np.array(body_points)
    indices = [5, 6, 12, 11]  # 肩膀和臀部的关键点
    square_points = keypoints[indices]

    # 计算原矩形区域的边界
    min_x = np.min(square_points[:, 0])
    max_x = np.max(square_points[:, 0])
    min_y = np.min(square_points[:, 1])
    max_y = np.max(square_points[:, 1])

    # 计算中心点
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 计算原始宽度和高度
    width = max_x - min_x
    height = max_y - min_y

    # 放大1.5倍
    new_width = width * 1.5
    new_height = height # * 1.5

    # 计算新的边界点
    new_min_x = max(0, center_x - new_width / 2)
    new_max_x = min(img_array.shape[1], center_x + new_width / 2)
    new_min_y = max(0, center_y - new_height / 2)
    new_max_y = min(img_array.shape[0], center_y + new_height / 2)

    # 裁剪图像
    cloth_img = img_array[int(new_min_y):int(new_max_y), int(new_min_x):int(new_max_x)]

    return cloth_img
