#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import sys
import time
import re
import csv
import traceback

# scene detection
import torch
sys.path.append('vidSum')
from src.utils import *
# descriptions / clustering / embeddings
from clusters import ClusterFrame
from embeddings import Embedder
from embeddings_dino import Embedder_dino  # 兼容原工程
import openai
import tiktoken

from src.model.model import myModel


# =========================
# QFVS Prompt（基于你提供的“原因”）
# =========================
def _qfvs_rubric_block() -> str:
    """
    完全基于用户提供的“原因”抽象出的 QFVS 评分规约（多查询打分版）。
    """
    return (
        "You are a segment evaluator for the QFVS (Query-Focused Video Summarization) dataset.\n"
        "Score ONLY the target scene using the global video description and the user queries; local context may be used if provided.\n\n"
        "Ground rules\n"
        "- Judge importance strictly by what is visible in the target scene: where we are, what task is pursued, who interacts with which objects, and what concrete, verifiable details are shown.\n"
        "- Static, generic, or off-topic shots are weak unless they add clear, new information that advances the main thread.\n"
        "- If the internal reasoning chain (rubric cues below) appears to conflict with a query's theme, ALWAYS prioritize the query theme for scoring; treat the reasoning chain as guidance, not as an override.\n\n"
        "### Multi-tier scoring rubric (0-100 total)\n"
        "1) Destination / Task Anchoring (0-30)\n"
        "   - Strong: explicit venue/task anchors (inside restaurant/bar; supermarket aisle; classroom with projector; workstation) and on-screen identifiers (menu/price board; book title or labeled artifact).\n"
        "   - Weak: floating outdoor laptop angles with indistinct text; ceiling lights/decor; street B-roll without tying to the task.\n\n"
        "2) Action -> Visible Outcome (0-20)\n"
        "   - Strong: concrete interactions that change the situation and are legible (drive that resolves inside venue; tray handling/ordering/drinking; consulting a paper list/receipt while selecting; cashier checkout; students raising hands; active typing/input).\n"
        "   - Weak: idle presence/ambience or repetitive motions that do not alter state (e.g., more eating with no new interaction; generic kitchen prep unrelated to dining/shopping thread).\n\n"
        "3) State Change & Narrative Progress (0-20)\n"
        "   - Strong: clear before->after or bridge (en-route->ready to order; outside->in-car->aisle->checkout; source materials->classroom engagement).\n"
        "   - Weak: remains in the same undifferentiated state; cutaways that do not advance the main thread (e.g., unrelated LEGO build, sidewalk landmark).\n\n"
        "4) Evidence & Identifiers (0-15)\n"
        "   - Strong: specific, readable cues that verify context/choices (menu items with prices, labeled book/artwork, paper list/receipt, dashboard/projection).\n"
        "   - Weak: partially visible/illegible content; hints without verifiable detail.\n\n"
        "5) Visual Clarity & Focus (0-15)\n"
        "   - Strong: key elements unobstructed and easy to parse (actor, object, text, transaction).\n"
        "   - Penalties for dark/blur/occluded framing that hides the important interaction.\n\n"
        "### Scenario checklists (apply only if consistent with the query)\n"
        "- Public Dining / Ordering: inside venue; tray/ordering/drink action; readable menu with prices.\n"
        "- Travel -> Venue: motion that resolves into destination interior (not just streets).\n"
        "- Shopping / Acquisition: aisle with paper list/receipt -> item selection -> cashier checkout.\n"
        "- Study / Lecture: named sources (book title, labeled sculpture) -> students watching/raising hands.\n"
        "- Tech / Workstation: laptop with active input (not just desktop wallpaper or vague outdoor angles).\n\n"
        "### Standard penalties (apply within the relevant dimension)\n"
        "- OFF_TOPIC cutaway (e.g., LEGO build, unrelated landmark, kitchen prep in a dining/shopping thread): -10\n"
        "- STATIC_AMBIENCE (lights/decor/idle patrons with no action): -8\n"
        "- REDUNDANT_REPEAT (near-duplicate of adjacent content without new info): -6\n"
        "- INDISTINCT_TEXT (menu/title/labels unreadable): -6\n"
        "- LOW_VISIBILITY (dark/blur/occluded key act): -6\n\n"
        "### Calibration ladder (use full scale; avoid bunching)\n"
        "- 90-100: Indispensable step/turning point with clear action, strong identifiers, obvious state change.\n"
        "- 75-89: Strong task action with minor gaps (e.g., inside venue with tray/drink but limited identifiers).\n"
        "- 60-74: Supportive context or partial step; anchors present but weaker change/evidence.\n"
        "- 40-59: Light context/ambience or unclear linkage to the task.\n"
        "- 0-39: Off-topic/static/illegible/redundant.\n\n"
        "### Query alignment modifier (±0-10 applied to Dimension 1)\n"
        "- Strong match to the stated query/topic -> +6 to +10\n"
        "- Clear contradiction/irrelevance -> -6 to -10\n"
        "Clamp final score to [0,100].\n\n"
        "### Mandatory internal calculation (do not reveal)\n"
        "Compute sub-scores in [0,100]: A=Anchoring, C=Causality(Action), S=StateChange, E=Evidence, V=Visibility.\n"
        "Apply penalties within affected dimensions. If target is near-duplicate of prev/next by wording/visuals, set E or S low and reduce uniqueness contribution accordingly.\n"
        "FINAL = round(0.30*A + 0.20*C + 0.20*S + 0.15*E + 0.15*V + PrefAdj), clamped to [0,100]. Use the full scale; typical scenes should land in 45-60 unless clearly strong/weak. Do not overuse a single value.\n\n"
        "### Output mode for MULTIPLE queries (IMPORTANT)\n"
        "- You will be given K user queries. For the ONE target scene, output K integers (0-100), ONE PER QUERY, strictly in the same order as provided.\n"
        "- For each query, if rubric cues and the query theme conflict, prioritize alignment with that query's theme; use rubric cues only as secondary guidance.\n"
        "- Output ONLY a single line: comma-separated integers, no extra text or spaces.\n"
        "- Example: 25,60,15,80\n"
    )



def _build_prompt_multiq(video_description: str,
                         part_description: str,
                         user_queries: list) -> str:
    """
    构造“多查询同场景”的评分提示词。
    关键：沿用上面的 QFVS 规约，并明确输出为“逗号分隔的一行 K 个整数”。
    """
    header = _qfvs_rubric_block()
    qsec = ""
    if user_queries:
        qsec += "User Queries (evaluate the scene AGAINST EACH query, and output one score per query in order):\n"
        for i, q in enumerate(user_queries, 1):
            qsec += f"Query {i}: {q}\n"
    return (
        f"{qsec}\n"
        f"{header}\n"
        f"Global Video Description:\n{video_description.strip()}\n\n"
        f"Target Scene Description:\n{part_description.strip()}\n\n"
        f"Return K comma-separated integers ONLY."
    )


# =========================
# OpenAI 调用（多查询一行结果）
# =========================
def _call_openai(prompt: str, model: str, timeout: int = 60, max_retries: int = 5) -> str:
    """
    带超时与退避重试的 ChatCompletion 调用。
    """
    backoff = [2, 4, 8, 15, 30]
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "request_timeout": timeout,
            }
            # 非 gpt-5 系列允许设温度为 0
            if not str(model).lower().startswith("gpt-5"):
                payload["temperature"] = 0.0

            resp = openai.ChatCompletion.create(**payload)
            return resp['choices'][0]['message']['content']
        except Exception as e:
            wait = backoff[min(attempt, len(backoff)-1)]
            print(f"⚠️ OpenAI call failed (attempt {attempt+1}/{max_retries}): {e} → retry in {wait}s", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)


def _parse_csv_line(line: str, k: int) -> list:
    """
    解析形如 '25,60,15,...' 的一行，返回长度为 k 的整数列表；
    解析失败则抛异常，让上层回退默认分数。
    """
    parts = [p.strip() for p in line.strip().split(',')]
    if len(parts) != k:
        raise ValueError(f"expected {k} scores, got {len(parts)}: {line}")
    out = []
    for p in parts:
        if not re.fullmatch(r"\d{1,3}", p):
            raise ValueError(f"non-integer token: {p}")
        v = max(0, min(100, int(p)))
        out.append(v)
    return out


# =========================
# 主流程：先检测已有文件 → 打分（多查询一行）
# =========================
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
        gpt_model = "gpt-5"  # 可改为 "gpt-5-mini" 节约成本

        # init embedder / clusterer
        embedder = Embedder(device)
        cluster_algo = ClusterFrame()

        # 先绑定当前视频（不带 VidQry），确保后续路径可用
        my_model.set_video_meta_data(video_name, None)

        # ===== 基本信息 =====
        print(f"[QFVS] device={device}", flush=True)
        print(f"[QFVS] video_name={video_name}", flush=True)
        print(f"[QFVS] video_path={my_model.video_path}", flush=True)

        # ===== 先做“已存在文件检测” =====
        scene_desc_dir = os.path.join(args.work_dir, "sceneDesc")
        os.makedirs(scene_desc_dir, exist_ok=True)
        scene_description_file = os.path.normpath(os.path.join(scene_desc_dir, f"{video_name}.json"))
        print(f"[QFVS] expect sceneDesc file: {scene_description_file}", flush=True)

        emb_base = os.path.normpath(my_model.frame_emb_file)      # 无扩展名基路径
        emb_npy  = emb_base + ".npy"
        print(f"[QFVS] expect frameEmb file: {emb_npy}", flush=True)

        # ---------- 场景检测（即使已有描述也需要 start_frames） ----------
        scene_list, start_frames = my_model.detect_scences_v2(None)
        print(f"[QFVS] detect_scences_v2: scenes={len(start_frames)-1} n_frames={my_model.n_frames}", flush=True)
        if start_frames and start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)

        # ---------- 抽帧 ----------
        frames = fetch_frames_v2(my_model.video_path)
        print(f"[QFVS] frames loaded: {len(frames)}", flush=True)

        # ---------- 特征：存在就复用，不存在才生成 ----------
        if os.path.exists(emb_npy):
            print("[QFVS] embeddings exist → reuse.", flush=True)
        else:
            print("[QFVS] caching frame embeddings ...", flush=True)
            embedder.cache_frame_embeddings(emb_base, frames)
            if os.path.exists(emb_npy):
                print("[QFVS] embeddings saved.", flush=True)
            else:
                print(f"[QFVS][ERR] embedding cache expected but file missing: {emb_npy}", flush=True)
                return 1

        # ---------- 合并场景 ----------
        start_frames = my_model.merge_scenes(start_frames, emb_npy, min_frames=MN_FRAMES)
        if start_frames and start_frames[-1] < my_model.n_frames:
            start_frames.append(my_model.n_frames)
        print(f"[QFVS] merge_scenes: scenes={len(start_frames)-1}", flush=True)

        # ---------- 描述文件：存在就用，不存在才生成 ----------
        if os.path.exists(scene_description_file):
            print("[QFVS] scene description exists → reuse.", flush=True)
        else:
            print("[QFVS] generating scene descriptions ...", flush=True)
            ret_path = my_model.generate_scene_descriptions(frames, start_frames)
            try:
                ret_path = os.path.normpath(ret_path) if ret_path else scene_description_file
            except Exception:
                ret_path = scene_description_file

            if os.path.exists(scene_description_file):
                pass
            elif ret_path and os.path.exists(ret_path):
                scene_description_file = ret_path
            else:
                print(f"[QFVS][ERR] scene description file missing: "
                      f"{scene_description_file} (fallback: {ret_path})", flush=True)
                return 2
            print(f"[QFVS] scene_desc_file ready at: {scene_description_file}", flush=True)

        # ---------- 一致性/不相似度 ----------
        frames_consistency, frames_dissimilarity = my_model.calc_frames_data(
            cluster_algo, start_frames, len(frames)
        )
        print("[QFVS] frames_consistency/dissimilarity computed.", flush=True)

        # ---------- 收集当前视频的所有 VidQry → user_queries 列表 ----------
        user_queries_map = []  # [(VidQry, query_string), ...]
        for VidQry in mapping.keys():
            if mapping[VidQry]['video_id'] != video_name:
                continue
            user_queries_map.append((VidQry, mapping[VidQry]['query']))
        if not user_queries_map:
            print("[QFVS][WARN] No queries for this video. Exit.", flush=True)
            return 0

        # 统一的 CSV 输出文件（与旧实现一致）
        csv_file = os.path.normpath(os.path.join(args.work_dir, f"{video_name}_output.csv"))
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['scene_num', 'output'])
        print(f"[QFVS] csv_file={csv_file}", flush=True)

        # ---------- 读取描述 JSON ----------
        with open(scene_description_file, "r") as jf:
            descriptions = json.load(jf)

        # 场景键：按 scene_1_description, scene_2_description ... 顺序
        scene_indices = []
        i = 1
        while f"scene_{i}_description" in descriptions:
            scene_indices.append(i)
            i += 1
        if not scene_indices:
            # 回退：按 keys 解析排序
            part_keys = [k for k in descriptions.keys()
                         if k.startswith('scene_') and k.endswith('_description')]
            part_keys.sort(key=lambda x: int(x.split('_')[1]))
            scene_indices = [int(k.split('_')[1]) for k in part_keys]
        num_scenes = len(scene_indices)
        print(f"[QFVS] scenes to score per query = {num_scenes}", flush=True)

        # ---------- 核心评分循环（多查询一行） ----------
        # 我们一次场景把 K 个 query 都塞进 prompt，要求模型返回 K 个分数（逗号分隔）
        # 然后把每个 VidQry 自己的分数序列积累起来（与旧实现一致）
        num_queries = len(user_queries_map)
        # 初始化：每个查询对应一个列表，长度=场景数
        query_scores = [[] for _ in range(num_queries)]

        for idx, scene_id in enumerate(scene_indices, 1):
            print(f"[QFVS] Processing Scene {scene_id} ({idx}/{num_scenes}) ...", flush=True)

            # 构造 prompt（多查询）
            video_desc = descriptions.get('video_description', '')
            part_desc  = descriptions.get(f'scene_{scene_id}_description', '')
            uq_list    = [q for _, q in user_queries_map]
            prompt     = _build_prompt_multiq(video_desc, part_desc, uq_list)

            #（可选）限速：简单 sleep 避免过快触发限流
            time.sleep(0.05)

            # 调用 OpenAI
            try:
                output_text = _call_openai(prompt, gpt_model, timeout=60, max_retries=5)
            except Exception as e:
                print(f"[QFVS][ERR] OpenAI failed on scene {scene_id}: {e} → use fallback 50s", flush=True)
                # 回退：给所有查询填 50
                for q_idx in range(num_queries):
                    query_scores[q_idx].append(50)
                # 写 CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([scene_id, "fallback_50"])
                continue

            print(f"[QFVS] raw output: {output_text}", flush=True)

            # 解析 “25,60,15,...” 为 K 个分数
            try:
                scene_scores = _parse_csv_line(output_text, num_queries)
            except Exception as e:
                print(f"[QFVS][ERR] parse error on scene {scene_id}: {e} → use fallback 50s", flush=True)
                scene_scores = [50] * num_queries

            # 累加到各查询的序列
            for q_idx, score in enumerate(scene_scores):
                query_scores[q_idx].append(score)

            # 记录到 CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scene_id, ",".join(str(s) for s in scene_scores)])

        print("[QFVS] scoring done for all scenes.", flush=True)

        # ---------- 把每个 VidQry 的分数保存为 meta ----------
        # 注意：query_scores 的顺序与 user_queries_map 一致
        for q_idx, (VidQry, user_query) in enumerate(user_queries_map):
            my_model.set_video_meta_data(video_name, VidQry)

            out_file = os.path.normpath(my_model.prediciton_meta_data_file)
            out_dir  = os.path.dirname(out_file)
            os.makedirs(out_dir, exist_ok=True)

            if os.path.exists(out_file):
                print(f"[QFVS] SKIP (meta exists): {VidQry} -> {out_file}", flush=True)
                continue

            print(f"[QFVS] saving results for {VidQry} -> {out_file}", flush=True)
            my_model.prediciton_meta_data_file = out_file
            my_model.save_results(
                query_scores[q_idx],     # 当前查询的整段场景分数
                start_frames,
                frames_consistency,
                frames_dissimilarity,
                user_query
            )
            print(f"[QFVS] {VidQry} DONE -> {out_file}", flush=True)

    except Exception as e:
        print(f"[FATAL] 未捕获异常：{e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 99


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAIN (QFVS: detect existing files; multi-query per scene scoring with latest prompt)")
    parser.add_argument("--openai_key", type=str, required=True)
    parser.add_argument("--video_name", type=str, choices=['P01', 'P02', 'P03', 'P04'], required=True)
    parser.add_argument("--video_dir", type=str, help="video/s directory")
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm'], default='mp4')
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--description_model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_scene_duration", type=int, default=2)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--segment_duration", type=int, default=1, help="Segment duration for processing")
    args = parser.parse_args()
    sys.exit(run(args))
