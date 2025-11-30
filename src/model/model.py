#utils
import argparse
import os
import json
# scene detection
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
# scores prediction
import numpy as np
import torch
import os
import sys 
import time
sys.path.append('vidSum')
from src.utils import *
# descriptions
from description_generator import DescriptionGenerator
import openai
from sklearn.metrics.pairwise import cosine_similarity
import csv
import tiktoken
import time
import cv2
from openai.error import RateLimitError
def generate_scene_prompt(video_description, part_description, user_query):
    prompt = ""
    prompt += "You are tasked with evaluating the importance of a specific scene within a larger video, considering its role in the overall narrative and message of the video. I've provided two descriptions below: one for the entire video and one for the specific scene (part) within that video.\n"
    prompt += "Your goal is to assess how critical this particular segment is to the understanding or development of the video's main themes, messages, or emotional impact.\n"
    if user_query != '':
        prompt += f"\nThe user has provided the following content preference to guide the summarization:\n"
        prompt += f"**User Query : {user_query}**\n"
        prompt += (
            "When assigning a score, consider how well the scene aligns with this preference. "
            "Scenes that closely match or contradict the user’s intent should be scored accordingly, "
            "reflecting their relevance or irrelevance to the desired summary focus.\n"
            "If the scene is not clearly related to this preference, assign a score based on the default scale and criteria below.\n"
        )

    prompt += "Assign an importance score on a scale of 1 to 100, based on how crucial it is to the overall video. The scale is defined as follows:\n"
    prompt += "* 1-20: Minimally important (contributes very little to the overall theme or message)\n"
    prompt += "* 21-40: Somewhat important (offers limited context or details that support the main theme)\n"
    prompt += "* 41-60: Moderately important (provides useful context or details that support the main theme)\n"
    prompt += "* 61-80: Quite important (adds significant context or detail that enhances understanding of the main theme)\n"
    prompt += "* 81-100: Highly important (crucial to understanding or conveying the main message of the video)\n"
    prompt += "When evaluating, focus on the core narrative or emotional impact of the video. Only assign high scores (80+) to the segments that **directly drive the main theme or message forward**. Be critical and biased towards giving low scores to segments that do not add significant value to the overall narrative or theme. The distribution of high scores should be low and reserved for only the most crucial moments in the video.\n"
    prompt += "The video should be summarized briefly, so please evaluate whether the scene is critical to include in the summary of the video, based on its contribution to the core message. **Prioritize scenes that are essential for a concise summary and omit secondary or supporting moments unless they provide meaningful context.**\n"
    prompt += "Provide only the score in your answer, without any explanation or reasoning.\n"

    prompt += "Whole Video Description: " + video_description + '\n'
    prompt += "Part Description: " + part_description + '\n'

    return prompt


def generate_scene_prompt_QFVS(video_description, part_description, user_queries):
    prompt = ""
    prompt += "You are tasked with evaluating the importance of a specific scene within a larger video, considering its role in the overall narrative and message of the video. I've provided two descriptions below: one for the entire video and one for the specific scene (part) within that video.\n"
    prompt += "Your goal is to assess how critical this particular segment is to the understanding or development of the video's main themes, messages, or emotional impact.\n"
    if len(user_queries) > 0:
        prompt += f"\nThe user has provided the following content preferences to guide the summarization:\n"
        prompt += f"**User Queries**:\n"
        for i, query in enumerate(user_queries, 1):
            prompt += f"Query {i} :  {query}\n"
        prompt += (
            "When assigning a score, consider how well the scene aligns with each query. "
            "Scenes that closely match or contradict the user’s intent should be scored accordingly, "
            "reflecting their relevance or irrelevance to the desired summary focus.\n"
        )
        prompt += "Assign a score representing the importance of the scene **for each user query**.\n"
        prompt += "If the scene is not clearly related to any of the queries, assign a score based on the default scale and criteria below.\n"

    prompt += "Assign an importance score on a scale of 1 to 100, based on how crucial it is to the overall video. The scale is defined as follows:\n"
    prompt += "* 1-20: Minimally important (contributes very little to the overall theme or message)\n"
    prompt += "* 21-40: Somewhat important (offers limited context or details that support the main theme)\n"
    prompt += "* 41-60: Moderately important (provides useful context or details that support the main theme)\n"
    prompt += "* 61-80: Quite important (adds significant context or detail that enhances understanding of the main theme)\n"
    prompt += "* 81-100: Highly important (crucial to understanding or conveying the main message of the video)\n"
    prompt += "When evaluating, focus on the core narrative or emotional impact of the video. Only assign high scores (80+) to the segments that **directly drive the main theme or message forward**. Be critical and biased towards giving low scores to segments that do not add significant value to the overall narrative or theme. The distribution of high scores should be low and reserved for only the most crucial moments in the video.\n"
    prompt += "The video should be summarized briefly, so please evaluate whether the scene is critical to include in the summary of the video, based on its contribution to the core message. **Prioritize scenes that are essential for a concise summary and omit secondary or supporting moments unless they provide meaningful context.**\n"
    prompt += "Provide only the score in your answer, without any explanation or reasoning.\n"
    prompt += "**Following this Format:**\n"
    prompt += "Return a single line with one importance score per query, separated by commas.\n"
    prompt += "For example: 25,60,15,80, ... \n"

    prompt += "Whole Video Description: " + video_description + '\n'
    prompt += "Part Description: " + part_description + '\n'

    return prompt

def generate_scene_prompt_sliding(video_description, part_descriptions, user_query):
    prompt = ""
    prompt += (
        "You are tasked with evaluating the **relative importance** of multiple scenes within a larger video.\n"
        "You will be given the full video description, and several short scene descriptions extracted from it.\n"
        "Your job is to assign each scene a score based on its contribution to the video's main message, emotional core, or narrative development.\n\n"
    )

    if user_query != '':
        prompt += f"The user has also provided a content preference to guide the summarization:\n"
        prompt += f"**User Query: {user_query}**\n"
        prompt += (
            "When assigning scores, prioritize scenes that are **directly relevant** to this user query. "
            "Penalize scenes that do not relate to the user's interest, even if they are part of the original story.\n\n"
        )

    # 思维链部分：引导模型逐步思考
    prompt += "Think step by step before you score:\n"
    prompt += (
        "1. First, **understand the overall theme or emotional core** of the video from the whole video description.\n"
        "2. Then, for each scene:\n"
        "   - Analyze what new information it contributes.\n"
        "   - Consider whether it introduces, develops, or concludes any major theme.\n"
        "   - Compare the scene with others: does it repeat content, or provide unique and necessary insight?\n"
        "   - Think about the **progression between scenes**: Does the current scene logically follow or contrast with the others?\n"
        "3. Focus on **how much value each scene adds** if we were to make a concise summary.\n"
        "4. Only assign high scores to the most crucial scenes. Be selective and avoid inflating all scores.\n"
    )

    # 打分标准说明
    prompt += "\nScore each scene on a scale of 1 to 100, using these levels:\n"
    prompt += "* 1–20: Minimally important\n"
    prompt += "* 21–40: Somewhat important\n"
    prompt += "* 41–60: Moderately important\n"
    prompt += "* 61–80: Quite important\n"
    prompt += "* 81–100: Highly important (only for key turning points or core message scenes)\n\n"

    # 输入内容
    prompt += "**Video Description:**\n"
    prompt += video_description.strip() + '\n\n'

    for i, part in enumerate(part_descriptions):
        prompt += f"**Scene {i + 1} Description:** {part.strip()}\n"

    prompt += (
        "\nNow, assign a score to **each scene** based on the above criteria.\n"
        "Output only the scores, comma-separated, in the order of the scenes above. Do not provide any explanation.\n"
        "Example: 45, 80, 30\n"
    )

    return prompt


def init_description_model(description_model_name):
    pretrained = description_model_name
    model_name = "llava_qwen"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = "auto"
    conv_model_template = "qwen_1_5"
    model = DescriptionGenerator(pretrained, model_name, conv_model_template, device=device, device_map=device_map)
    return model


class myModel:

    def __init__(self, args):

        self.video_dir = args.video_dir
        self.work_dir = args.work_dir
        
        self.video_type = args.video_type
        self.video_fps = None
        self.selected_sst = None
        self.n_frames = None
        self.prediciton_meta_data_file = None
        self.min_scene_duration = args.min_scene_duration
        self.window_size = args.segment_duration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scene_description_dir = self.work_dir + '/sceneDesc'
        self.prediciton_meta_data_dir = self.work_dir + '/PredMetaData'
        self.frame_emb_dir = self.work_dir + '/FrameEmb/'

        self.description_model = init_description_model(args.description_model_name)
        self.batch_size = args.batch_size


        self.scale = 100
        self.meta_data = {}
        self.token_count = 0
        self.TPM = 200_000
        self.gpt_model = "gpt-4o"

    #
    def set_video_meta_data(self, video_name, VidQry):
        self.VidQry = VidQry
        self.video_name = video_name
        self.video_path = self.video_dir + '/' + self.video_name + '.' + self.video_type
        self.n_frames = get_video_frames_num(self.video_path)
        if self.VidQry != '' :
            self.prediciton_meta_data_file = self.prediciton_meta_data_dir + f'/{self.VidQry}.json'
        else:
            self.prediciton_meta_data_file = self.prediciton_meta_data_dir + f'/{self.video_name}.json'
        self.frame_emb_file = self.frame_emb_dir + f'/{self.video_name}'
        return
    
    #
    def init_pipline_dirs(self, dirs_names):
        for dir_name in dirs_names:
            dir_path = self.work_dir + '/' + dir_name
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)  # Create the directory
                except Exception as e:
                    raise ValueError(f"Error creating directory '{dir_path}': {e}")

    # 
    def detected_scenes_num(self, sst):
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=sst, min_scene_len=self.min_scene_duration*self.video_fps))
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        return len(scene_list)

    # select threhold 
    def scene_threshold_selection(self, sst_range=range(2,71,2), history_n=8):
        lens = []
        acc_range = []
        for h, sst in enumerate(sst_range):
            curr_scene_n = self.detected_scenes_num(sst)
            # early stoping 
            if h >= history_n and np.all([curr_scene_n == x for x in lens[h-history_n:h]]):
                break
            lens.append(curr_scene_n)
            acc_range.append(sst)
            # Plot the second video in the pair if available
        # 
        lens = np.array(lens)
        lens_diff = lens[1:] - lens[:-1]
        mn = np.argmin(lens_diff)
        selected_sst = (acc_range[mn+1] + acc_range[mn])/2
        return int(selected_sst)

    # detect scences
    def detect_scences(self, sst=None):

        self.video_fps = get_video_FPS(self.video_path)
        if sst == None :
            self.selected_sst = self.scene_threshold_selection()
        else :
            self.selected_sst = sst
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.selected_sst, min_scene_len=self.min_scene_duration*self.video_fps))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # List of scenes with start and end timecodes
        scene_list = scene_manager.get_scene_list()
        if len(scene_list) > 0:
            frames_start = [int(self.video_fps * scene[0].get_seconds()) for scene in scene_list]
            return [[scene[0], scene[1]] for scene in scene_list], frames_start
        
        self.selected_sst = 2
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.selected_sst, min_scene_len=self.min_scene_duration*self.video_fps))

        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # List of scenes with start and end timecodes
        scene_list = scene_manager.get_scene_list()
        frames_start = [int(self.video_fps * scene[0].get_seconds()) for scene in scene_list]

        return [[scene[0], scene[1]] for scene in scene_list], frames_start

    def detect_scences_v2(self, sst=None):

        # ===== 1) 基本信息 =====
        self.video_fps = get_video_FPS(self.video_path)
        SHOT_SECONDS = 5  # 5 秒步长
        print(f"[DEBUG-detect] 开始场景检测：视频FPS={self.video_fps}，抽帧步长={SHOT_SECONDS}秒", flush=True)

        # 阈值选择逻辑保持不变
        if sst is None:
            self.selected_sst = self.scene_threshold_selection()
        else:
            self.selected_sst = sst
        print(f"[DEBUG-detect] 选定场景检测阈值：{self.selected_sst}", flush=True)

        # ===== 2) 初始化 =====
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()

        # 采样后“有效帧率”（每 5 秒只看一帧）
        frames_per_step = max(1, int(round(self.video_fps * SHOT_SECONDS)))
        frame_skip = frames_per_step - 1
        effective_fps = self.video_fps / (frame_skip + 1)
        print(f"[DEBUG-detect] 抽帧配置：原始帧步长={frames_per_step}帧/5秒，有效FPS={effective_fps:.2f}", flush=True)

        # 最短场景长度：换算到采样后帧数
        min_scene_len_frames = max(2, int(round(self.min_scene_duration * effective_fps)))
        print(f"[DEBUG-detect] 最短场景长度：{min_scene_len_frames}个抽帧窗口", flush=True)

        scene_manager.add_detector(
            ContentDetector(
                threshold=self.selected_sst,
                min_scene_len=min_scene_len_frames
            )
        )

        # ===== 3) 检测（每 5 秒采样一次）=====
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager, frame_skip=frame_skip)

        # ===== 4) 处理场景结果 =====
        scene_list = scene_manager.get_scene_list()
        if len(scene_list) > 0:
            # 用 5 秒抽帧后的 shot 索引生成 frames_start
            frames_start = [int(scene[0].get_seconds() / SHOT_SECONDS) for scene in scene_list]
            print(
                f"[DEBUG-detect] 场景检测原始结果：{len(scene_list)}个场景，frames_start原始首尾={frames_start[0]}~{frames_start[-1]}",
                flush=True)

            # 关键修复1：n_frames 设为“最后一个抽帧索引 + 1”（代表抽帧总数量）
            self.n_frames = frames_start[-1] + 1 if frames_start else 0
            print(f"[DEBUG-detect] 计算抽帧总数量：self.n_frames={self.n_frames}（最后抽帧索引+1）", flush=True)

            # 关键修复2：确保 frames_start 从0开始，且以 n_frames 结尾
            if frames_start[0] != 0:
                print(f"[DEBUG-detect] 补全frames_start开头：插入0（原开头为{frames_start[0]}）", flush=True)
                frames_start.insert(0, 0)
            if frames_start[-1] < self.n_frames:
                print(f"[DEBUG-detect] 补全frames_start结尾：插入{self.n_frames}（原结尾为{frames_start[-1]}）",
                      flush=True)
                frames_start.append(self.n_frames)

            # 最终验证日志：输出补全后的完整信息
            print(
                f"[DEBUG-detect] 最终场景结果：{len(frames_start) - 1}个场景，frames_start首尾={frames_start[0]}~{frames_start[-1]}，抽帧总数量={self.n_frames}",
                flush=True)
            return [[scene[0], scene[1]] for scene in scene_list], frames_start

        # ===== 5) 阈值降到 2 再试 =====
        print(f"[DEBUG-detect] 初始检测无场景，阈值降至2重试", flush=True)
        self.selected_sst = 2
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.selected_sst,
                min_scene_len=min_scene_len_frames
            )
        )
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager, frame_skip=frame_skip)

        # ===== 6) 处理阈值调整后的场景结果 =====
        scene_list = scene_manager.get_scene_list()
        frames_start = [int(scene[0].get_seconds() / SHOT_SECONDS) for scene in scene_list] if scene_list else []
        print(
            f"[DEBUG-detect] 阈值调整后原始结果：{len(scene_list)}个场景，frames_start原始首尾={frames_start[0] if frames_start else '无'}~{frames_start[-1] if frames_start else '无'}",
            flush=True)

        # 同样修复 n_frames 和 frames_start
        self.n_frames = frames_start[-1] + 1 if (frames_start and len(frames_start) > 0) else 0
        print(f"[DEBUG-detect] 计算抽帧总数量：self.n_frames={self.n_frames}（最后抽帧索引+1）", flush=True)

        if frames_start and frames_start[0] != 0:
            print(f"[DEBUG-detect] 补全frames_start开头：插入0（原开头为{frames_start[0]}）", flush=True)
            frames_start.insert(0, 0)
        if frames_start and frames_start[-1] < self.n_frames:
            print(f"[DEBUG-detect] 补全frames_start结尾：插入{self.n_frames}（原结尾为{frames_start[-1]}）", flush=True)
            frames_start.append(self.n_frames)

        # 最终验证日志：输出补全后的完整信息
        print(
            f"[DEBUG-detect] 最终场景结果：{len(frames_start) - 1 if frames_start else 0}个场景，frames_start首尾={frames_start[0] if frames_start else '无'}~{frames_start[-1] if frames_start else '无'}，抽帧总数量={self.n_frames}",
            flush=True)
        return [[scene[0], scene[1]] for scene in scene_list], frames_start

    #
    def merge_scenes(self, scenes_bounds_frames, frame_emb_file, min_frames):
        """
        Merges small scenes based on cosine similarity with neighboring scenes.

        Args:
            scenes_bounds_frames (list): List of scene start frame numbers.
            frame_emb_file (str): File path to the .npy file containing frame embeddings.
            min_frames (int): Minimum number of frames a scene must have to avoid merging.

        Returns:
            list: New list of scene boundaries after merging.
        """
        framesEmd_file = np.load(frame_emb_file, allow_pickle=True)

        while True:
            # Compute the average embedding for each scene
            scene_embeddings = []
            scene_lengths = []
            valid_scenes = []  # Store the scene start frames and their corresponding end frames

            for i in range(len(scenes_bounds_frames) - 1):
                start, end = scenes_bounds_frames[i], scenes_bounds_frames[i + 1]
                frames = framesEmd_file[start:end]  # Directly slice the numpy array
                if frames.size > 0:
                    mean_embedding = np.mean(frames, axis=0)
                else:
                    mean_embedding = np.zeros_like(framesEmd_file[0])  # Ensure valid shape (using the first frame's embedding)

                scene_embeddings.append(mean_embedding)
                scene_lengths.append(end - start)
                valid_scenes.append((start, end))  # Store the scene's start and end

            merged = False
            new_scenes = []  # List to store the updated scene boundaries
            i = 0

            while i < len(scene_embeddings):
                if scene_lengths[i] >= min_frames:
                    new_scenes.append(valid_scenes[i][1])  # Keep the current scene if it's large enough
                    i += 1
                    continue

                # Compute cosine similarity with previous and next scenes (if they exist)
                prev_sim = cosine_similarity(scene_embeddings[i].reshape(1, -1), scene_embeddings[i - 1].reshape(1, -1))[0][0] if i > 0 else -1
                next_sim = cosine_similarity(scene_embeddings[i].reshape(1, -1), scene_embeddings[i + 1].reshape(1, -1))[0][0] if i < len(scene_embeddings) - 1 else -1

                # Merge with the more similar scene
                if prev_sim > next_sim and i > 0:
                    # Merge with the previous scene
                    scene_embeddings[i - 1] = (scene_embeddings[i - 1] * scene_lengths[i - 1] + scene_embeddings[i] * scene_lengths[i]) / (scene_lengths[i - 1] + scene_lengths[i])
                    scene_lengths[i - 1] += scene_lengths[i]
                    merged = True  # A merge occurred
                    i += 1  # Skip the current scene since it's merged with the previous one
                elif next_sim >= prev_sim and i < len(scene_embeddings) - 1:
                    # Merge with the next scene
                    scene_embeddings[i + 1] = (scene_embeddings[i + 1] * scene_lengths[i + 1] + scene_embeddings[i] * scene_lengths[i]) / (scene_lengths[i + 1] + scene_lengths[i])
                    scene_lengths[i + 1] += scene_lengths[i]
                    new_scenes.append(valid_scenes[i + 1][1])  # Corrected to append end frame of merged scene
                    merged = True  # A merge occurred
                    i += 1  # Skip the current scene since it's merged with the next one
                else:
                    # If no merge, keep the current scene
                    new_scenes.append(valid_scenes[i][1])
                    i += 1

            # If no scenes were merged, exit the loop
            if not merged:
                break

            # Update the valid scenes after merging
            scenes_bounds_frames = [valid_scenes[0][0]] + new_scenes  # Update with the new boundaries


        if scenes_bounds_frames[-1] < self.n_frames:
            scenes_bounds_frames.append(self.n_frames)
        while scenes_bounds_frames[-1] - scenes_bounds_frames[-2] < min_frames:
            scenes_bounds_frames.pop(-2)
        while scenes_bounds_frames[1] < min_frames:
            scenes_bounds_frames.pop(1)


        return scenes_bounds_frames

    # # generate scenes description
    # def generate_scene_descriptions(self, video_frames, scene_frames):
    #
    #     file_name = self.scene_description_dir + '/' + self.video_name + '.json'
    #     if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
    #         return file_name
    #
    #     scene_descriptions = {}
    #     all_video_description = self.description_model.generate_description_batch_frames_v2(video_frames, self.video_fps, 1, self.batch_size).replace('\n',' ')
    #
    #     scene_descriptions['video_description'] = all_video_description
    #     for i, (start, end) in enumerate(zip(scene_frames[:-1], scene_frames[1:])):
    #         print(f'Generating description for  : {start} - {end}')
    #         part_description = self.description_model.generate_description_batch_frames_v2(video_frames[start:end], self.video_fps, 2, self.batch_size).replace('\n',' ')
    #         scene_descriptions[f"scene_{i+1}_description"] = part_description
    #
    #
    #     with open(file_name, 'w') as json_file:
    #         json.dump(scene_descriptions, json_file, indent=4)
    #
    #     return file_name
    # generate scenes description
    def generate_scene_descriptions(self, video_frames, scene_frames):
        file_name = self.scene_description_dir + '/' + self.video_name + '.json'

        # 如果已有文件则直接返回
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            return file_name

        scene_descriptions = {}

        # 全视频描述
        all_video_description = self.description_model.generate_description_batch_frames_v2(
            video_frames, self.video_fps, 1, self.batch_size
        ).replace('\n', ' ')
        print(f"[Full Video Description]: {all_video_description}\n")
        scene_descriptions['video_description'] = all_video_description

        # 每段 scene 的描述
        for i, (start, end) in enumerate(zip(scene_frames[:-1], scene_frames[1:])):
            print(f'Generating description for scene {i + 1} ({start} - {end})...')
            part_description = self.description_model.generate_description_batch_frames_v2(
                video_frames[start:end], self.video_fps, 2, self.batch_size
            ).replace('\n', ' ')
            print(f'Description for scene {i + 1}: {part_description}\n')
            scene_descriptions[f"scene_{i + 1}_description"] = part_description

        # 写入 JSON 文件
        with open(file_name, 'w') as json_file:
            json.dump(scene_descriptions, json_file, indent=4)

        return file_name

    # scene score
    # def compute_scenes_score(self, discription_file_name, user_query):
    #     # load descriptions
    #     descriptions = []
    #     with open(discription_file_name, "r") as json_file:
    #         descriptions = json.load(json_file)
    #
    #     scores = []
    #     for i in range(1,len(descriptions.keys())):
    #
    #         input_text = generate_scene_prompt(descriptions['video_description'], descriptions[f'scene_{i}_description'], user_query)
    #         input_size = len(input_text.split(' '))
    #
    #         if(self.token_count  + input_size > self.TPM):
    #             print('sleep')
    #             time.sleep(60)
    #             self.token_count = 0
    #         response = openai.ChatCompletion.create(
    #         model=self.gpt_model,
    #         messages=[
    #             {"role": "user", "content": input_text}
    #         ], temperature=0.5)
    #         #
    #         output_text = response['choices'][0]['message']['content']
    #         print(output_text)
    #         scene_score = int(output_text)
    #         scores.append(scene_score)
    #
    #         self.token_count += input_size
    #
    #
    #     return scores

    def compute_scenes_score(self, discription_file_name, user_query):

        tokenizer = tiktoken.encoding_for_model(self.gpt_model)

        with open(discription_file_name, "r") as json_file:
            descriptions = json.load(json_file)

        scores = []
        for i in range(1, len(descriptions.keys())):
            input_text = generate_scene_prompt(
                descriptions['video_description'],
                descriptions[f'scene_{i}_description'],
                user_query
            )
            input_size = len(tokenizer.encode(input_text))

            # ✅ 本地提前判断（可保留）
            if self.token_count + input_size > self.TPM:
                print('💤 token count approaching limit, sleeping 60s...')
                time.sleep(60)
                self.token_count = 0

            # ✅ 尝试调用 + 出错后 sleep 重试
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_model,
                        messages=[{"role": "user", "content": input_text}],
                        temperature=0.5
                    )
                    break  # 成功跳出循环
                except RateLimitError as e:
                    print(f"⛔ Rate limit hit: {e}. Sleeping 10s and retrying...")
                    time.sleep(10)

            output_text = response['choices'][0]['message']['content']
            print(output_text)

            try:
                scene_score = int(output_text)
            except ValueError:
                print("⚠️ GPT output is not an int, using 0 instead")
                scene_score = 0

            scores.append(scene_score)

            output_size = len(tokenizer.encode(output_text))
            self.token_count += input_size + output_size

        return scores

    def compute_scenes_slide_score(self, discription_file_name, user_query, window_size=3, stride=1):
        tokenizer = tiktoken.encoding_for_model(self.gpt_model)

        with open(discription_file_name, "r") as json_file:
            descriptions = json.load(json_file)

        part_keys = [k for k in descriptions.keys() if k.startswith('scene_') and k.endswith('_description')]
        part_keys.sort(key=lambda x: int(x.split('_')[1]))  # sort scene_1, scene_2, ...

        num_scenes = len(part_keys)
        scene_scores = [[] for _ in range(num_scenes)]  # for averaging later

        for start in range(0, num_scenes - window_size + 1, stride):
            part_descriptions = [
                descriptions[part_keys[start + i]] for i in range(window_size)
            ]

            input_text = generate_scene_prompt_sliding(
                descriptions['video_description'],
                part_descriptions,
                user_query
            )

            input_size = len(tokenizer.encode(input_text))
            if self.token_count + input_size > self.TPM:
                print('token count approaching limit, sleeping 60s...')
                time.sleep(60)
                self.token_count = 0

            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_model,
                        messages=[{"role": "user", "content": input_text}],
                        temperature=0.5
                    )
                    break
                except openai.error.RateLimitError as e:
                    print(f"⛔ Rate limit hit: {e}. Sleeping 10s and retrying...")
                    time.sleep(10)

            output_text = response['choices'][0]['message']['content']
            print(f"GPT Output (scenes {start + 1}-{start + window_size}): {output_text}")

            try:
                scores = [int(s.strip()) for s in output_text.strip().split(',')]
                assert len(scores) == window_size
            except Exception as e:
                print(f"⚠️ Parsing failed for GPT output: {output_text}. Using zeros.")
                scores = [0] * window_size

            for i in range(window_size):
                scene_scores[start + i].append(scores[i])

            output_size = len(tokenizer.encode(output_text))
            self.token_count += input_size + output_size

        # 平均每个场景的分数
        final_scores = []
        for i, score_list in enumerate(scene_scores):
            if score_list:
                avg_score = int(sum(score_list) / len(score_list))
            else:
                avg_score = 0
            final_scores.append(avg_score)

        return final_scores

    #
    def compute_scenes_score_QFVS(self, discription_file_name, user_queries):
        # Load descriptions
        with open(discription_file_name, "r") as json_file:
            descriptions = json.load(json_file)

        csv_file = f'{self.work_dir}/{self.video_name}_output.csv'
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['scene_num', 'output'])
                
        num_queries = len(user_queries)
        query_scores = [[] for _ in range(num_queries)]
        i = 1
        while i < len(descriptions.keys()):
            print(f'Processing Scene {i} ... ')
            input_text = generate_scene_prompt_QFVS(
                descriptions['video_description'],
                descriptions[f'scene_{i}_description'],
                user_queries
            )

            input_size = len(input_text.split())

            if self.token_count + input_size > self.TPM:
                print("Sleeping for rate limit")
                time.sleep(60)
                self.token_count = 0

            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": input_text}],
                temperature=0.5,
            )
            output_text = response['choices'][0]['message']['content']
            print(output_text)

            # Parse response: assume response is like "34, 15, 87, ..."
            try :
                scene_scores = [int(s.strip()) for s in output_text.split(',')]
            except:
                self.token_count += input_size
                continue

            if len(scene_scores) != num_queries :
                continue
            
            for q_idx, score in enumerate(scene_scores):
                query_scores[q_idx].append(score)

            # Append new rows
            csv_file = f'{self.work_dir}/{self.video_name}_output.csv'
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i, output_text])
            i += 1

        # read from csv
        return query_scores  #num_queries X num_scenes
    
    # segment score 
    def calc_frames_data(self, cluster_algo, start_frames, n_frames):
        frames_consistency = []
        frames_dissimilarity = []
        scene_start_frames = start_frames
        if scene_start_frames[-1] < n_frames:
            scene_start_frames.append(n_frames)

        embeddings = np.load(self.frame_emb_file + '.npy')
        
        segments_consistency = []
        segments_dissimilarity = []
        for i, (start, end) in enumerate(zip(scene_start_frames[:-1], scene_start_frames[1:])):
            
            scene_embeddings = embeddings[start : end]
            labels = cluster_algo.automate_clustering(scene_embeddings)
            segments_labels, segment_indcies   = cluster_algo.segment_labels(labels, end-start, self.window_size * self.video_fps)
            
            for segment_labels, segment_index in zip(segments_labels, segment_indcies):
                segment_embeddings = embeddings[start + segment_index[0]: start + segment_index[1]]
                consistency_score, dissimilarity_score = cluster_algo.segment_contribution(segment_labels, segment_embeddings)
                segments_consistency.append((consistency_score, len(segment_labels)))
                segments_dissimilarity.append((dissimilarity_score, len(segment_labels)))

        for score, n_segment in segments_consistency:
            for _ in range(n_segment):
                frames_consistency.append(score)


        for score, n_segment in segments_dissimilarity:
            for _ in range(n_segment):
                frames_dissimilarity.append(score)
         
        assert len(frames_consistency) == n_frames, 'Error in casting segments to frames in Frame Scoring'
        assert len(frames_dissimilarity) == n_frames, 'Error in casting segments to frames in Frame Scoring'

        return frames_consistency, frames_dissimilarity

    # save video meta data 
    def save_results(self, scene_scores, scene_frames, frames_consistency, frames_dissimilarity, user_query):
        
        video_prediction_meta_data = {}
        video_prediction_meta_data['video_name'] = self.video_name
        video_prediction_meta_data['query'] = user_query
        video_prediction_meta_data['video_path'] = self.video_path
        video_prediction_meta_data['video_fps'] = self.video_fps
        video_prediction_meta_data['scene_scores'] = scene_scores
        video_prediction_meta_data['scene_frames'] = scene_frames
        video_prediction_meta_data['n_frames'] = self.n_frames
        video_prediction_meta_data['consistency'] = frames_consistency
        video_prediction_meta_data['dissimilarity'] = frames_dissimilarity
        video_prediction_meta_data['sst'] = self.selected_sst
        
        # cache video's embeddings 
        with open(self.prediciton_meta_data_file, 'w') as json_file:
            json.dump(video_prediction_meta_data, json_file, indent=4)
        print(self.prediciton_meta_data_file)
