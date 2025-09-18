# # #
# This module interacts with mediapipe 
# for recording or processing prediction videos
# or preprocessing and encoding training videos 
# # #
import mediapipe as mp
import os
import cv2
from torchvision import transforms
import numpy as np 
import time
from tqdm import tqdm

KEYPOINTS_SIZE = 231
FRAME_COUNT = 139

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
])
mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) #holistic model
mp_drawing = mp.solutions.drawing_utils 

def preprocess_video(input_path):
    cap = cv2.VideoCapture(input_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    frames_multiplier = 30 / fps
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(framecount * frames_multiplier)

def process_video(input_path, max_frames):
    cap = cv2.VideoCapture(input_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    repeat_or_skip_freq = 100000000
    if fps < 30:
        #repeat every nth frame to simulate 30 fps
        repeat_or_skip_freq = int(round(1 / (30 / fps - 1)))
    if fps > 30:
        #skip every nth frame to simulate 30 fps
        repeat_or_skip_freq = int(round(30 / (fps - 30))) + 1

    
    repeat_counter = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        repeat_counter += 1
        if (fps > 30 and repeat_or_skip_freq == repeat_counter):
            repeat_counter = 0
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = extract_keypoints(mp_holistic.process(frame))
        frames.append(np.asarray(keypoints, dtype=np.float32))

        if (repeat_or_skip_freq == repeat_counter):
            repeat_counter = 0
            frames.append(np.asarray(keypoints, dtype=np.float32))

    cap.release()
    pad = np.asarray(np.zeros(KEYPOINTS_SIZE), dtype=np.float32)
    while (len(frames) < max_frames):
        frames.append(pad)
    return frames

def save_video(frames, output_path_base):
    output = open(output_path_base + ".piped", 'wb')
    for frame in frames:
        frame.tofile(output)
    output.close()
    output = open(output_path_base + "_mirrored.piped", 'wb')
    for frame in frames:
        y = np.asarray([-frame[i] if i % 3 == 0 else frame[i] for i in range(len(frame))])  
        y.tofile(output)   
    output.close()

def record_and_process_video():
    frames = []
    cap = cv2.VideoCapture(0) #default bgr need rgb
    #set mediapipe model
    while cap.isOpened():
        ret, frame = cap.read()
        
        frames.append(np.copy(frame))
        if FRAME_COUNT < len(frames):
            frames.pop(0)
        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(1000 // 30) != -1:
            break
    
    for i in range(len(frames)):
        start = time.time()
        res = mp_holistic.process(frames[i])
        draw_landmarks(frames[i], res)
        cv2.imshow('OpenCV Feed', frames[i])
        frames[i] = extract_keypoints(res)
        elapsed = time.time() - start
        cv2.waitKey(max(round(1000 * (1 / 30 - elapsed)), 1))
    while (len(frames) < FRAME_COUNT):
        frames.append(np.zeros(KEYPOINTS_SIZE))
        
    cap.release()
    cv2.destroyAllWindows()
    return frames

def centralize_and_normalize_landmark(landmark):
    # for each column (x,y, or z), normalize to a standard distribution
    landmark = landmark - landmark.mean(0)
    st = landmark.std(0)
    for i in range(3):
        if abs(st[i]) < 0.0000001:
            st[i] = 1
    return landmark / st

def face_AABB(face_points):
    # AABB stands for axis-aligned bounding box
    # Row 0 is min, row 1 is max
    aabb = np.zeros((2,3))
    for i in range(face_points.shape[0]):
        aabb[0] = np.min(np.stack([aabb[0], face_points[i]], 0), axis=0)
        aabb[1] = np.max(np.stack([aabb[1], face_points[i]], 0), axis=0)
    return aabb
        

def extract_keypoints(result):
    pose = np.array([[res.x, res.y, res.z] for res in result.pose_landmarks.landmark]) if result.pose_landmarks else np.zeros((33,3))
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]) if result.face_landmarks else np.zeros((468,3))
    left_hand = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]) if result.left_hand_landmarks else np.zeros((21,3))
    right_hand = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]) if result.right_hand_landmarks else np.zeros((21,3))
    everything = np.concatenate((face_AABB(face), pose, left_hand, right_hand), axis=0)
    
    return centralize_and_normalize_landmark(everything).flatten()

def draw_landmarks(image, res):
    landmark_spec = mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1)
    connection_spec = mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    mp_drawing.draw_landmarks(image,res.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION, landmark_spec, connection_spec)
    mp_drawing.draw_landmarks(image,res.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS, landmark_spec, connection_spec)
    mp_drawing.draw_landmarks(image,res.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, landmark_spec, connection_spec)
    mp_drawing.draw_landmarks(image,res.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS, landmark_spec, connection_spec)


def pipe_videos():
    max_frames = 0
    vids = []
    data_dir = 'asl_translator/src/data/daniela'
    output_dir = 'asl_translator/src/data/piped_std_naomi'
    for class_name in sorted(os.listdir(data_dir)):     
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
        class_dir = os.path.join(data_dir, class_name)
        for video_file in os.listdir(class_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                noext, _ = os.path.splitext(video_file)
                input_path = os.path.join(class_dir, video_file)
                output_path_base = os.path.join(output_dir, class_name, noext)
                if (os.path.exists(output_path_base + ".piped") and os.path.exists(output_path_base + "_mirrored.piped")):
                    continue
                vids.append((video_file, class_name))

    prog_bar = tqdm(vids, desc="Prep")
    for (video_file, class_name) in prog_bar:
        input_path = os.path.join(data_dir, class_name, video_file)
        framecount = preprocess_video(input_path)
        if max_frames < framecount:
            max_frames = framecount
        prog_bar.set_postfix({'max_fr': max_frames})
    print("max frames:", max_frames)
    if (max_frames != FRAME_COUNT):
        print(f"Changing max_frames to {FRAME_COUNT}. Please ensure that this is the correct value.")
    max_frames = FRAME_COUNT
    prog_bar = tqdm(vids, desc="Proc")
    for (video_file, class_name) in prog_bar:
        noext, _ = os.path.splitext(video_file)
        input_path = os.path.join(data_dir, class_name, video_file)
        output_path_base = os.path.join(output_dir, class_name, noext)
        prog_bar.set_postfix({'filename': noext})
        save_video(process_video(input_path, max_frames), output_path_base)
                    

if __name__ == "__main__":
    pipe_videos()


