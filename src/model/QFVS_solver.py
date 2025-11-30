import argparse
import os
import json
import sys
import traceback

# scene detection
import torch
from src.utils import *
# descriptions
from clusters import ClusterFrame
from embeddings import Embedder
from embeddings_dino import Embedder_dino
import openai
from src.model.model import myModel


def run(args):
    try:
        openai.api_key = args.openai_key

        video_name = args.video_name
        mapping_file = args.mapping_file

        with open(mapping_file, 'r') as json_file:
            mapping = json.load(json_file)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        my_model = myModel(args)
        my_model.init_pipline_dirs(['sceneDesc', 'FrameEmb'])

        MN_FRAMES = 150

        # init embedder / clusterer
        embedder = Embedder(device)
        cluster_algo = ClusterFrame()

        # 先绑定当前视频（不带 VidQry），确保后续路径可用
        my_model.set_video_meta_data(video_name, None)

        # ===== 关键定位打印（不改逻辑）=====
        print(f"[QFVS] device={device}", flush=True)
        print(f"[QFVS] video_name={video_name}", flush=True)
        print(f"[QFVS] video_path={my_model.video_path}", flush=True)
        #print(f"[QFVS] frame_emb_file(no ext)={my_model.frame_emb_file}", flush=True)

        # ---------- scene detection ----------
        scene_list, start_frames = my_model.detect_scences_v2(None)
        print(f"[QFVS] detect_scences: scenes={len(start_frames)-1} n_frames={my_model.n_frames}", flush=True)

        if start_frames and start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        # ---------- cache frame embeddings ----------
        frames = fetch_frames_v2(my_model.video_path)
        print(f"[QFVS] frames loaded: {len(frames)}", flush=True)

        # === 修复点 1：路径规范化 ===
        emb_npy = os.path.normpath(my_model.frame_emb_file + '.npy')
        print(f"[QFVS] emb_npy={emb_npy}", flush=True)

        if not os.path.exists(emb_npy):
            print("[QFVS] caching frame embeddings ...", flush=True)
            embedder.cache_frame_embeddings(os.path.normpath(my_model.frame_emb_file), frames)

            # 再次规范化检查
            if os.path.exists(emb_npy):
                print("[QFVS] embeddings saved.", flush=True)
            else:
                print(f"[QFVS][ERR] embedding cache expected but file missing: {emb_npy}", flush=True)
                return 1
        else:
            print("[QFVS] embeddings exist, reuse.", flush=True)

        # ---------- merge scenes ----------
        start_frames = my_model.merge_scenes(start_frames, emb_npy, min_frames=MN_FRAMES)
        if start_frames and start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)
        print(f"[QFVS] merge_scenes: scenes={len(start_frames)-1}", flush=True)

        # ---------- generate scene descriptions ----------
        scene_discription_file_name = my_model.generate_scene_descriptions(frames, start_frames)
        print(f"[QFVS] scene_desc_file={scene_discription_file_name}", flush=True)

        # === 修复点 2：文件路径安全检查 ===
        try:
            scene_discription_file_name = os.path.normpath(scene_discription_file_name)
            desc_ok = bool(scene_discription_file_name) and os.path.exists(scene_discription_file_name)
        except Exception:
            desc_ok = False
        if not desc_ok:
            print("[QFVS][ERR] scene description file missing/invalid. "
                  "Check generate_scene_descriptions() internals/requirements.", flush=True)
            return 2

        # ---------- frames consistency / dissimilarity ----------
        frames_consistency, frames_dissimilarity = my_model.calc_frames_data(
            cluster_algo, start_frames, len(frames)
        )
        print("[QFVS] frames_consistency/dissimilarity computed.", flush=True)

        # ---------- collect queries ----------
        user_queries = []
        run_vidqrs = []  # 与 queries 对齐，便于打印
        for VidQry in mapping.keys():
            if mapping[VidQry]['video_id'] != video_name:
                continue

            torch.cuda.empty_cache()
            my_model.set_video_meta_data(video_name, VidQry)

            # === 修复点 3：预测文件路径规范化 ===
            pred_file = os.path.normpath(my_model.prediciton_meta_data_file)
            if os.path.exists(pred_file):
                print(f"[QFVS] SKIP (meta exists): {VidQry} -> {pred_file}", flush=True)
                continue

            print(f"[QFVS] will run VidQry={VidQry}", flush=True)
            user_queries.append(mapping[VidQry]['query'])
            run_vidqrs.append(VidQry)

        print(f"[QFVS] queries_to_run={len(user_queries)}", flush=True)
        if not user_queries:
            print("[QFVS][WARN] No queries to run (mapping empty for this video OR all skipped). Exit.", flush=True)
            return 0

        # ---------- scoring ----------
        print("[QFVS] Predicting ...", flush=True)
        queries_scores = my_model.compute_scenes_score_QFVS(scene_discription_file_name, user_queries)

        # 基本健壮性检查（只打印，不改变原有计算）
        if not isinstance(queries_scores, (list, tuple)) or len(queries_scores) != len(user_queries):
            l = (len(queries_scores) if isinstance(queries_scores, (list, tuple)) else "NA")
            print(f"[QFVS][ERR] compute_scenes_score_QFVS returned invalid result. len(scores)={l}, "
                  f"len(queries)={len(user_queries)}", flush=True)
            return 3

        print("[QFVS] scoring done.", flush=True)

        # ---------- save results ----------
        for i, VidQry in enumerate(run_vidqrs):
            user_query = user_queries[i]
            scene_scores = queries_scores[i]

            # 设置 meta 信息（保证路径正确）
            my_model.set_video_meta_data(video_name, VidQry)

            # 路径规范化
            out_file = os.path.normpath(my_model.prediciton_meta_data_file)
            out_dir = os.path.dirname(out_file)
            os.makedirs(out_dir, exist_ok=True)  # ← 保证父目录存在

            if os.path.exists(out_file):
                print(f"[QFVS] SKIP (meta exists): {VidQry} -> {out_file}", flush=True)
                continue

            print(f"[QFVS] saving results for {VidQry} -> {out_file}", flush=True)

            # 保证保存的路径和上面一致
            my_model.prediciton_meta_data_file = out_file
            my_model.save_results(scene_scores, start_frames, frames_consistency, frames_dissimilarity, user_query)

            print(f"[QFVS] {VidQry} DONE -> {out_file}", flush=True)

    except Exception as e:
        print(f"[FATAL] 未捕获异常：{e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 99


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAIN")
    parser.add_argument("--openai_key", type=str)
    parser.add_argument("--video_name", type=str, choices=['P01', 'P02', 'P03', 'P04'])
    parser.add_argument("--video_dir", type=str, help="video/s directory")
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm'], default='mp4')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--description_model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_scene_duration", type=int, default=2)
    parser.add_argument("--mapping_file", type=str)
    parser.add_argument("--segment_duration", type=int, default=1, help="Segment duration for processing")
    args = parser.parse_args()
    sys.exit(run(args))

