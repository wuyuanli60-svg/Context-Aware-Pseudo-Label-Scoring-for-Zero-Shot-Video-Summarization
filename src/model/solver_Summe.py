# utils
import argparse
import os
import json
# scene detection
import torch
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
import os
import sys
import time

sys.path.append('vidSum')
from src.utils import *
# descriptions
from clusters import ClusterFrame
from embeddings import Embedder
from embeddings_dino import Embedder_dino
import openai
from src.model.model import myModel


def solve(args):
    openai.api_key = args.openai_key
    if args.video_name == '':
        videos_names = list(os.listdir(args.video_dir))
        videos_names = [name.split('.')[0] for name in videos_names if name.endswith('.' + args.video_type)]
    else:
        videos_names = [args.video_name]

    user_query = args.user_query
    VidQry = args.VidQry

    if user_query != '':
        assert VidQry != '', 'If user query is provided you should pass a key for the pair (video, query), e.g. VidQry_{i}.'

    if user_query == '':
        assert VidQry == '', 'Key for the pair (video, query) doesn\'t isn\'t supported when the query isn\'t provided.'

    device = "cuda"
    my_model = myModel(args)
    my_model.init_pipline_dirs(['sceneDesc', 'FrameEmb'])

    # init embedder
    embedder = Embedder(device)
    # init Clusterer
    cluster_algo = ClusterFrame()

    for video_name in videos_names:
        my_model.set_video_meta_data(video_name, VidQry)

        # === ADD: 判断 PredMetaData_1/{video}.json 是否已存在 ===
        pred_file_path = os.path.join(args.work_dir, "PredMetaData_1", f"{video_name}.json")
        if os.path.exists(pred_file_path):
            print(f"⚠️ {video_name} 已存在 {pred_file_path}，跳过处理。")
            continue
        # === END ADD ===

        torch.cuda.empty_cache()

        # scene detection
        scene_list, start_frames = my_model.detect_scences(None)
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        # cache frame embeddings
        frames = fetch_frames(my_model.video_path)
        if not os.path.exists(my_model.frame_emb_file + '.npy'):
            embedder.cache_frame_embeddings(my_model.frame_emb_file, frames)

        start_frames = my_model.merge_scenes(start_frames, my_model.frame_emb_file + '.npy', min_frames=150)

        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        # 1. 生成 scene 描述
        scene_discription_file_name = my_model.generate_scene_descriptions(frames, start_frames)

        # 2. 计算场景分数
        scene_scores = my_model.compute_scenes_score(scene_discription_file_name, user_query)

        # 3. 保存不同窗口大小结果
        for W in range(1, 5):
            my_model.window_size = W
            meta_dir = f'PredMetaData_{W}'
            my_model.init_pipline_dirs([meta_dir])
            my_model.prediciton_meta_data_dir = os.path.join(my_model.work_dir, meta_dir)
            my_model.set_video_meta_data(video_name, user_query)

            frames_consistency, frames_dissimilarity = my_model.calc_frames_data(
                cluster_algo, start_frames, len(frames)
            )

            my_model.save_results(
                scene_scores, start_frames, frames_consistency, frames_dissimilarity, user_query
            )

        print(f"✅ {video_name} DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAIN")
    parser.add_argument("--video_name", type=str, default='')
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm', 'avi', 'mov', 'mkv'],
                        help='video extension (e.g., mp4)', default='mp4')
    parser.add_argument("--user_query", type=str, default='')
    parser.add_argument("--VidQry", type=str, default='', help='Key for the (video, query) pair - results directory')
    parser.add_argument("--video_dir", type=str, help="video/s directory", required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--description_model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_scene_duration", type=int, default=2)
    parser.add_argument("--segment_duration", type=int, default=1)
    parser.add_argument("--openai_key", type=str, required=True)

    args = parser.parse_args()
    solve(args)
