import cv2

def write_to_file_with_limit(text, file, max_words_per_line = 25):
    words = text.replace("\n","").split()
    for i in range(0, len(words), max_words_per_line):
        line = " ".join(words[i:i+max_words_per_line])
        file.write(line + '\n')

def get_video_FPS(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # cast float fps to the nearset int number
    video_fps = int(fps + 0.5)
    return video_fps

def get_video_frames_num(video_path):
    cap = cv2.VideoCapture(video_path)
    n = 0
    while True :
        ret, _ = cap.read()
        if not ret:
            break
        n += 1

    cap.release()
    return n
def fetch_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames

def fetch_frames_v2(video_path):
    """
    每 5 秒采样一帧（取该时间段的中点帧），保证与 detect_scences(5s) 对齐。
    返回帧列表：frames[i] 对应第 i 个 5 秒 shot。
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video meta: fps={fps}, total_frames={total_frames}")

    # 视频总时长（秒）
    duration = total_frames / fps
    shot_seconds = 5.0  # 与 detect_scences 保持一致

    frames = []
    t = 0.0
    while t < duration - 1e-6:
        # 当前 5 秒窗口的中点时间
        mid_t = min(t + shot_seconds / 2.0, duration - 1e-6)
        frame_idx = int(round(mid_t * fps))
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        t += shot_seconds

    cap.release()
    return frames

def read_non_overlapping_segments(video_path, segment_duration, video_fps):
    cap = cv2.VideoCapture(video_path)
    frames_per_segment = int(segment_duration * video_fps)
    segments = []
    while True:
        segment = []
        for _ in range(frames_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
            segment.append(frame)
        if not segment:
            break
        segments.append(segment)
        # Skip to the next segment
        #cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_per_segment)
    cap.release()
    return segments, frames_per_segment

def read_non_overlapping_scenes(video_path, video_data):
    cap = cv2.VideoCapture(video_path)
    scenes = []
    scene_num = 0
    start_frames = video_data['scene_frames']
    if video_data['scene_frames'][-1] < video_data['n_frames']:
        start_frames += [video_data['n_frames']]
   
    while True:
        scene = []
        ret, frame = cap.read()
        if not ret:
            break
        for _ in range(start_frames[scene_num], start_frames[scene_num+1]):
            scene.append(frame)
            ret, frame = cap.read()
            if not ret:
                break
        if not scene:
            break
        scenes.append(scene)
        scene_num += 1
    cap.release()
    return scenes