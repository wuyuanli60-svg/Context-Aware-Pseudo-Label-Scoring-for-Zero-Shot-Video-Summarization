#utils
import argparse
import os
import json
# scene detection
import torch
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
        assert VidQry != '', 'If user query is provided you should pass a key for the pair (video, query), e.g. VidQry_\{i\}.'
    
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
        # if os.path.exists(my_model.prediciton_meta_data_file):
        #     if QUERY_PROVIDED :
        #         print(f'Meta Data for the Video-Query (w\ VidQry : {VidQry}) already exites, SKIPING!')
        #     else :
        #         print(f'Meta Data for the Video {video_name} already exites, SKIPING!')
        #     continue
        #
        # print(f'{video_name} Starting ...')
        # 使用绝对路径检查 PredMetaData_1 中是否已存在评分结果
        pred_file_path = os.path.join(
            "/usr1/home/s124mdg43_11/ZeroShotVideoSummary-main/tvsum_metadata",
            "PredMetaData_1",
            f"{video_name}.json"
        )

        # 如果存在就跳过整个流程
        if os.path.exists(pred_file_path):
            print(f"Prediction already exists for {video_name}, skipping full processing.")
            continue
        torch.cuda.empty_cache()
        
        #scene detection
        scene_list, start_frames = my_model.detect_scences(None)
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        #cache frame embeddings 
        frames = fetch_frames(my_model.video_path)
        if not os.path.exists(my_model.frame_emb_file + '.npy') :
            embedder.cache_frame_embeddings(my_model.frame_emb_file, frames)

        start_frames = my_model.merge_scenes(start_frames, my_model.frame_emb_file + '.npy', min_frames=150)
        
        if start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)


        #否则，开始完整流程
        # 1. 生成 scene 描述（内部有存在性判断和保存，不重复写入）
        scene_discription_file_name = my_model.generate_scene_descriptions(frames, start_frames)

        # 2. 计算场景分数（调用 OpenAI，大量消耗 token）
        scene_scores = my_model.compute_scenes_score(scene_discription_file_name, user_query)

        # 3. 对不同的窗口大小，分别保存结果（包括一致性和不相似性信息）
        for W in range(1, 5):
            my_model.window_size = W
            meta_dir = f'PredMetaData_{W}'
            my_model.init_pipline_dirs([meta_dir])
            my_model.prediciton_meta_data_dir = os.path.join(my_model.work_dir, meta_dir)
            my_model.set_video_meta_data(video_name, user_query)

            # 计算帧一致性与不相似性
            frames_consistency, frames_dissimilarity = my_model.calc_frames_data(
                cluster_algo, start_frames, len(frames)
            )

            # 保存结果（包括分数、mask 等元信息）
            my_model.save_results(
                scene_scores, start_frames, frames_consistency, frames_dissimilarity, user_query
            )

        print(f"✅ {video_name} DONE!")


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="MAIN")
    # Add arguments
    
    parser.add_argument("--video_name", type=str)
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm'], help='mp4 or webm')

    parser.add_argument("--user_query",type=str, default='')
    parser.add_argument("--VidQry", type=str, default='', help='Key for the (video, query) pair - results directory')
    parser.add_argument("--video_dir", type=str, help="video/s directory")
    
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--description_model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_scene_duration", type=int, default=2)
    parser.add_argument("--segment_duration", type=int, default=1)
    parser.add_argument("--openai_key", type=str)

    args = parser.parse_args()
    solve(args)