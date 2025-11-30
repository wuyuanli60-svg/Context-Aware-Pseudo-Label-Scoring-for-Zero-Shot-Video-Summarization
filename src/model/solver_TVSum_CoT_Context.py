import os
import json
import re
import torch
import sys
import time
import numpy as np
import tiktoken
import openai
import argparse
from src.utils import fetch_frames
from embeddings import Embedder
from clusters import ClusterFrame
from src.model.model import myModel


# =========================
# Prompt builders (tvsum_v2)
# =========================

def _tvsum_rubric_block():
    """Core rubric text for tvsum_v2 (score_only) + anti-collapse constraints."""
    return (
        "You are a segment evaluator for the TVSum dataset.\n"
        "Score ONLY the target scene using the global video description (and optional local context).\n\n"
        "### Multi-tier scoring rubric (0–100 total)\n"
        "1) Task/Thematic Relevance (0–35)\n"
        "   - Advances the core activity (hands-on steps: grooming/feeding/repair/cooking; riders in motion; performers/floats). Low if generic or unrelated.\n"
        "2) Action/Interaction & Skill (0–20)\n"
        "   - Concrete interactions (human–animal, human–tool, performer–crowd, rider–bike), visible skill/risk/teamwork.\n"
        "3) Detail & Visibility (0–15)\n"
        "   - Close-ups of hands/tools/animals/ingredients/parts; clarity to learn the step; hygiene/safety cues add credibility.\n"
        "4) Informational Uniqueness (0–15)\n"
        "   - New step/angle/outcome vs. adjacent scenes; penalize near-duplicates.\n"
        "5) Narrative Progression (0–15)\n"
        "   - Bridges setup→action→result or marks a turning point/outcome verification.\n\n"
        "### Category checklists (quick boosters)\n"
        "- PET_CARE: hands/tools near animal; calm control; before→during→after continuity.\n"
        "- DOG_SHOW: ring setting; handler–dog gait/stance; judge proximity; breed showcasing.\n"
        "- VEHICLE_REPAIR: tool handling; compressor/sealant; wheel fit/inspection; driving B-roll is weak.\n"
        "- CYCLING / BIKE_MAINT: riders in motion; lanes/peloton; for maintenance—hands-on tuning.\n"
        "- PARKOUR / STUNTS: jump/landing clarity; protective gear; booth talk is weak.\n"
        "- PARADE / FESTIVAL: performers/floats/choreography; title cards/empty streets are weak.\n"
        "- COOKING: prep/mix/heat/plating; color theme OK only if tied to a real step.\n"
        "- EDU_NATURE: structural close-ups; behavior/process visuals; unrelated species is weak.\n"
        "- INDUSTRIAL/LOGISTICS: vehicle doing the task (navigate/load/crane) > parked exteriors.\n\n"
        "### Standard penalties (apply within dimensions)\n"
        "- TITLE/LOGO/BLANK or promo-only card: −15\n"
        "- OFF_TOPIC: −10\n"
        "- STATIC_GENERIC (no task/motion/interaction): −8\n"
        "- LOW_VISIBILITY (dark/blur/occluded): −6\n"
        "- REDUNDANT_REPEAT (near-duplicate): −6\n\n"
        "### Calibration ladder (combat score inflation)\n"
        "- 90–100: Indispensable core step/climax with clear action & close-up detail; unique; advances outcome.\n"
        "- 75–89: Strong task action/performance; minor gaps.\n"
        "- 60–74: Supportive context or partial step with reasonable clarity.\n"
        "- 40–59: Light context/idle/decorative; weak linkage to task.\n"
        "- 0–39: Logo/promo/off-topic/static/poor visibility/redundant.\n\n"
        "### User preference modifier (±0–10 to Relevance; clamp 0–100)\n"
        "- Strong alignment with provided preference → +6~+10\n"
        "- Clear contradiction/irrelevance → −6~−10\n\n"
        "### Mandatory internal calculation (do not reveal)\n"
        "Compute sub-scores in [0,100]: R=Relevance, A=Action, D=Detail, U=Uniqueness, N=Narrative.\n"
        "Apply penalties within each dimension. If target ≈ prev/next by wording, set U ≤ 5.\n"
        "FINAL = round(0.35*R + 0.20*A + 0.15*D + 0.15*U + 0.15*N + PrefAdj); clamp to [0,100].\n"
        "Use the full scale; typical scenes should land in 45–60 unless clearly strong/weak.\n"
        "Do not overuse a single value across scenes.\n\n"
        "### Output mode\n"
        "- Output EXACTLY ONE integer in [0,100] (no text, no units)."
    )


def generate_prompt_target_only(video_description: str,
                                target_scene_text: str,
                                user_query: str) -> str:
    """
    Prompt for the first or last scene (no local context).
    Returns a score_only instruction; model must return one integer.
    """
    header = _tvsum_rubric_block()
    q = ""
    if user_query.strip():
        q = (
            f"\n\nUser preference (optional, may be empty): {user_query.strip()}\n"
            "Apply the preference modifier to the Relevance dimension if clearly aligned or misaligned."
        )
    prompt = (
        f"{header}\n\n"
        f"### Inputs\n"
        f"- Global video description:\n{video_description.strip()}\n\n"
        f"- Target scene description:\n{target_scene_text.strip()}\n"
        f"{q}\n\n"
        f"Score only the target scene. Return one integer."
    )
    return prompt


def generate_prompt_with_context(video_description: str,
                                 prev_scene_text: str,
                                 target_scene_text: str,
                                 next_scene_text: str,
                                 user_query: str) -> str:
    """
    Middle scenes with local context.
    Focus scoring primarily on the TARGET scene description and the Global video description.
    Use prev/next only for a small ±5 context adjustment. Output ONE integer 0–100.
    """
    header = _tvsum_rubric_block()

    q = ""
    if user_query.strip():
        q = (
            f"\n\nUser preference (optional): {user_query.strip()}\n"
            "Apply ONLY as a subtle modifier to Theme/Relevance (±5 at most) if clear alignment/misalignment exists; "
            "do not affect other dimensions."
        )

    # —— 简洁的上下文处理与限幅规则
    refine_block = (
        "### Context handling (read carefully; do this INTERNALLY before scoring)\n"
        "- Briefly REFINE the Previous and Next scenes SEPARATELY (1–2 short notes each). "
        "Keep ONLY key facts: who is involved, what happens (key action), stage/transition (setup/key/aftermath), "
        "any NEW or REPEATED information relative to the TARGET, and whether the key entity is clearly visible. "
        "Ignore decorative/background details. Do NOT reveal these notes.\n"
        "- Scoring priority: Derive the BASE score PRIMARILY from the TARGET scene description and the Global video description, "
        "strictly following the rubric dimensions.\n"
        "- Small context adjustment (±5):\n"
        "  + Add up to +5 ONLY if the TARGET clearly introduces NEW information/stage and avoids repetition vs BOTH neighbors.\n"
        "  – Subtract up to -5 ONLY if the TARGET is largely DUPLICATED/redundant and adds no meaningful progress vs BOTH neighbors.\n"
        "  0 If evidence is unclear or mixed, apply NO adjustment (0). This adjustment is small and conservative.\n"
        "- Always SCORE ONLY THE TARGET scene; prev/next are reference for the small ±5 adjustment, not direct scoring evidence."
    )

    # —— 输入顺序：突出“主要关注目标+全文”；prev/next 仅作参考
    prompt = (
        f"{header}\n\n"
        f"{refine_block}\n\n"
        f"### Inputs (focus on TARGET + Global; use prev/next only for the small ±5 adjustment)\n"
        f"- TARGET scene (score this one ONLY):\n{target_scene_text.strip()}\n\n"
        f"- Global video description:\n{video_description.strip()}\n\n"
        f"- Previous scene (context only):\n{prev_scene_text.strip()}\n\n"
        f"- Next scene (context only):\n{next_scene_text.strip()}\n"
        f"{q}\n\n"
        f"Return ONE integer 0–100 (no words, no units)."
    )
    return prompt
def generate_prompt_target_only(video_description: str,
                                target_scene_text: str,
                                user_query: str) -> str:
    """
    Prompt for the first or last scene (no local context).
    Returns a score_only instruction; model must return one integer.
    """
    header = _tvsum_rubric_block()
    q = ""
    if user_query.strip():
        q = (
            f"\n\nUser preference (optional, may be empty): {user_query.strip()}\n"
            "Apply the preference modifier to the Relevance dimension if clearly aligned or misaligned."
        )
    prompt = (
        f"{header}\n\n"
        f"### Inputs\n"
        f"- Global video description:\n{video_description.strip()}\n\n"
        f"- Target scene description:\n{target_scene_text.strip()}\n"
        f"{q}\n\n"
        f"Score only the target scene. Return one integer."
    )
    return prompt


def generate_prompt_with_context(video_description: str,
                                 prev_scene_text: str,
                                 target_scene_text: str,
                                 next_scene_text: str,
                                 user_query: str) -> str:
    """
    Middle scenes with local context.
    Focus scoring primarily on the TARGET scene description and the Global video description.
    Use prev/next only for a small ±5 context adjustment. Output ONE integer 0–100.
    """
    header = _tvsum_rubric_block()

    q = ""
    if user_query.strip():
        q = (
            f"\n\nUser preference (optional): {user_query.strip()}\n"
            "Apply ONLY as a subtle modifier to Theme/Relevance (±5 at most) if clear alignment/misalignment exists; "
            "do not affect other dimensions."
        )

    # —— 简洁的上下文处理与限幅规则
    refine_block = (
        "### Context handling (read carefully; do this INTERNALLY before scoring)\n"
        "- Briefly REFINE the Previous and Next scenes SEPARATELY (1–2 short notes each). "
        "Keep ONLY key facts: who is involved, what happens (key action), stage/transition (setup/key/aftermath), "
        "any NEW or REPEATED information relative to the TARGET, and whether the key entity is clearly visible. "
        "Ignore decorative/background details. Do NOT reveal these notes.\n"
        "- Scoring priority: Derive the BASE score PRIMARILY from the TARGET scene description and the Global video description, "
        "strictly following the rubric dimensions.\n"
        "- Small context adjustment (±5):\n"
        "  + Add up to +5 ONLY if the TARGET clearly introduces NEW information/stage and avoids repetition vs BOTH neighbors.\n"
        "  – Subtract up to -5 ONLY if the TARGET is largely DUPLICATED/redundant and adds no meaningful progress vs BOTH neighbors.\n"
        "  0 If evidence is unclear or mixed, apply NO adjustment (0). This adjustment is small and conservative.\n"
        "- Always SCORE ONLY THE TARGET scene; prev/next are reference for the small ±5 adjustment, not direct scoring evidence."
    )

    # —— 输入顺序：突出“主要关注目标+全文”；prev/next 仅作参考
    prompt = (
        f"{header}\n\n"
        f"{refine_block}\n\n"
        f"### Inputs (focus on TARGET + Global; use prev/next only for the small ±5 adjustment)\n"
        f"- TARGET scene (score this one ONLY):\n{target_scene_text.strip()}\n\n"
        f"- Global video description:\n{video_description.strip()}\n\n"
        f"- Previous scene (context only):\n{prev_scene_text.strip()}\n\n"
        f"- Next scene (context only):\n{next_scene_text.strip()}\n"
        f"{q}\n\n"
        f"Return ONE integer 0–100 (no words, no units)."
    )
    return prompt
def generate_prompt_target_only(video_description: str,
                                target_scene_text: str,
                                user_query: str) -> str:
    """
    Prompt for the first or last scene (no local context).
    Returns a score_only instruction; model must return one integer.
    """
    header = _tvsum_rubric_block()
    q = ""
    if user_query.strip():
        q = (
            f"\n\nUser preference (optional, may be empty): {user_query.strip()}\n"
            "Apply the preference modifier to the Relevance dimension if clearly aligned or misaligned."
        )
    prompt = (
        f"{header}\n\n"
        f"### Inputs\n"
        f"- Global video description:\n{video_description.strip()}\n\n"
        f"- Target scene description:\n{target_scene_text.strip()}\n"
        f"{q}\n\n"
        f"Score only the target scene. Return one integer."
    )
    return prompt


def generate_prompt_with_context(video_description: str,
                                 prev_scene_text: str,
                                 target_scene_text: str,
                                 next_scene_text: str,
                                 user_query: str) -> str:
    """
    Middle scenes with local context.
    Focus scoring primarily on the TARGET scene description and the Global video description.
    Use prev/next only for a small ±5 context adjustment. Output ONE integer 0–100.
    """
    header = _tvsum_rubric_block()

    q = ""
    if user_query.strip():
        q = (
            f"\n\nUser preference (optional): {user_query.strip()}\n"
            "Apply ONLY as a subtle modifier to Theme/Relevance (±5 at most) if clear alignment/misalignment exists; "
            "do not affect other dimensions."
        )

    # —— 简洁的上下文处理与限幅规则
    refine_block = (
        "### Context handling (read carefully; do this INTERNALLY before scoring)\n"
        "- Briefly REFINE the Previous and Next scenes SEPARATELY (1–2 short notes each). "
        "Keep ONLY key facts: who is involved, what happens (key action), stage/transition (setup/key/aftermath), "
        "any NEW or REPEATED information relative to the TARGET, and whether the key entity is clearly visible. "
        "Ignore decorative/background details. Do NOT reveal these notes.\n"
        "- Scoring priority: Derive the BASE score PRIMARILY from the TARGET scene description and the Global video description, "
        "strictly following the rubric dimensions.\n"
        "- Small context adjustment (±5):\n"
        "  + Add up to +5 ONLY if the TARGET clearly introduces NEW information/stage and avoids repetition vs BOTH neighbors.\n"
        "  – Subtract up to -5 ONLY if the TARGET is largely DUPLICATED/redundant and adds no meaningful progress vs BOTH neighbors.\n"
        "  0 If evidence is unclear or mixed, apply NO adjustment (0). This adjustment is small and conservative.\n"
        "- Always SCORE ONLY THE TARGET scene; prev/next are reference for the small ±5 adjustment, not direct scoring evidence."
    )

    # —— 输入顺序：突出“主要关注目标+全文”；prev/next 仅作参考
    prompt = (
        f"{header}\n\n"
        f"{refine_block}\n\n"
        f"### Inputs (focus on TARGET + Global; use prev/next only for the small ±5 adjustment)\n"
        f"- TARGET scene (score this one ONLY):\n{target_scene_text.strip()}\n\n"
        f"- Global video description:\n{video_description.strip()}\n\n"
        f"- Previous scene (context only):\n{prev_scene_text.strip()}\n\n"
        f"- Next scene (context only):\n{next_scene_text.strip()}\n"
        f"{q}\n\n"
        f"Return ONE integer 0–100 (no words, no units)."
    )
    return prompt



# =====================================
# Pipeline with scorer rewiring (GPT-5)
# =====================================

class SceneScorer:
    def __init__(self, gpt_model="gpt-5", TPM=120000):
        self.gpt_model = gpt_model
        self.TPM = TPM
        self.token_count = 0
        self.last_request_time = time.time()
        # gpt-5 可能不在旧 tiktoken 映射里，提供回退编码
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.gpt_model)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def parse_single_score(self, output_text: str) -> int:
        """
        解析单个整数分数（0-100），鲁棒回退为 50。
        """
        try:
            nums = re.findall(r'\b\d{1,3}\b', output_text)
            if not nums:
                return 50
            val = int(nums[0])
            if val < 0:
                return 0
            if val > 100:
                return 100
            return val
        except Exception as e:
            print(f"⚠️ Error parsing single score from: {output_text[:120]}... ({e}) -> use 50")
            return 50

    def _get_gpt_response(self, input_text):
        """
        调用 Chat Completions；openai==0.28.x 仅需 model + messages。
        对 GPT-5 不传 temperature（只支持默认1）；其他模型仍设为 0.0。
        """
        input_size = len(self.tokenizer.encode(input_text))

        # 每分钟滑窗重置令牌计数
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last > 60:
            self.token_count = 0
            self.last_request_time = current_time

        # 简单 TPM 控制
        if (self.token_count + input_size) > self.TPM:
            sleep_time = max(60 - time_since_last, 5)
            print(f'⏳ Token limit approaching ({self.token_count}/{self.TPM}), sleeping {sleep_time:.1f}s...')
            time.sleep(sleep_time)
            self.token_count = 0
            self.last_request_time = time.time()

        # 重试机制
        max_retries = 5
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.gpt_model,
                    "messages": [{"role": "user", "content": input_text}],
                }
                # 仅非 GPT-5 系列才传 temperature
                if not str(self.gpt_model).lower().startswith("gpt-5"):
                    payload["temperature"] = 0.0

                response = openai.ChatCompletion.create(**payload)
                output_text = response['choices'][0]['message']['content']
                output_size = len(self.tokenizer.encode(output_text))
                self.token_count += input_size + output_size
                return response
            except openai.error.RateLimitError as e:
                wait_time = min(10 * (attempt + 1), 60)
                print(f"⚠️ Rate limit (attempt {attempt + 1}/{max_retries}): {e} → sleep {wait_time}s")
                time.sleep(wait_time)
            except Exception as e:
                print(f"⚠️ Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise RuntimeError(f"GPT request failed after {max_retries} attempts: {e}")
        raise RuntimeError(f"GPT request failed after {max_retries} attempts")

    def compute_scenes_score(self, description_file_name: str, user_query: str):
        """
        新版核心评分函数：
        - 逐个片段打分（单次调用只返回一个整数）
        - 第一个/最后一个片段：无上下文 prompt
        - 中间片段：加入 prev/next 文本，但强调只评分 target
        """
        try:
            print(f"📖 Loading scene descriptions from: {description_file_name}")
            with open(description_file_name, "r") as json_file:
                descriptions = json.load(json_file)
        except Exception as e:
            print(f"❌ Error loading scene descriptions: {e}")
            return None

        # 收集片段文本
        part_keys = [k for k in descriptions.keys()
                     if k.startswith('scene_') and k.endswith('_description')]
        part_keys.sort(key=lambda x: int(x.split('_')[1]))
        num_scenes = len(part_keys)

        if num_scenes == 0:
            print("⚠️ No scene descriptions found in the file")
            return None

        print(f"🔍 Found {num_scenes} scenes for scoring (tvsum_v2, target-only with optional context)")
        video_desc = descriptions.get('video_description', '').strip()

        final_scores = []

        for idx in range(num_scenes):
            target_text = descriptions[part_keys[idx]]
            # 构造 Prompt
            if idx == 0 or idx == num_scenes - 1:
                # 首尾：无上下文
                prompt = generate_prompt_target_only(
                    video_description=video_desc,
                    target_scene_text=target_text,
                    user_query=user_query or ""
                )
                dbg_ctx = "no-context"
            else:
                # 中间：带前后上下文，但只评分中间
                prev_text = descriptions[part_keys[idx - 1]]
                next_text = descriptions[part_keys[idx + 1]]
                prompt = generate_prompt_with_context(
                    video_description=video_desc,
                    prev_scene_text=prev_text,
                    target_scene_text=target_text,
                    next_scene_text=next_text,
                    user_query=user_query or ""
                )
                dbg_ctx = "with-context"

            print(f"🧠 Scoring scene {idx + 1}/{num_scenes} ({dbg_ctx})...")
            try:
                resp = self._get_gpt_response(prompt)
                output_text = resp['choices'][0]['message']['content']
                score = self.parse_single_score(output_text)
                final_scores.append(score)
                print(f"🎯 Scene {idx + 1} score: {score}")
            except Exception as e:
                print(f"❌ Error generating score for scene {idx + 1}: {e} → use 50")
                final_scores.append(50)

        return final_scores


def solve(args):
    # 确保工作目录正确
    args.work_dir = os.path.abspath(args.work_dir)
    if args.output_suffix and not args.work_dir.endswith(args.output_suffix):
        args.work_dir = f"{args.work_dir.rstrip('_')}_{args.output_suffix}"

    print(f"🛠 Using work directory: {args.work_dir}")
    os.makedirs(args.work_dir, exist_ok=True)

    # 子目录
    scene_desc_dir = os.path.join(args.work_dir, "sceneDesc")
    frame_emb_dir = os.path.join(args.work_dir, "FrameEmb")
    os.makedirs(scene_desc_dir, exist_ok=True)
    os.makedirs(frame_emb_dir, exist_ok=True)
    print(f"📂 Created scene description directory: {scene_desc_dir}")
    print(f"📂 Created frame embeddings directory: {frame_emb_dir}")

    # PredMetaData 1-4
    for i in range(1, 5):
        pred_dir = os.path.join(args.work_dir, f"PredMetaData_{i}")
        os.makedirs(pred_dir, exist_ok=True)
        print(f"📂 Created prediction metadata directory: {pred_dir}")

    openai.api_key = args.openai_key
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Using device: {device}")

    my_model = myModel(args)
    embedder = Embedder(device)
    cluster_algo = ClusterFrame()
    # 默认使用 GPT-5。如需更便宜可换 "gpt-5-mini"
    scorer = SceneScorer(gpt_model="gpt-5")

    # 获取视频列表
    videos = [args.video_name] if args.video_name else [
        f.split('.')[0] for f in os.listdir(args.video_dir)
        if f.endswith('.' + args.video_type)
    ]
    print(f"🎬 Found {len(videos)} videos to process")

    for video_name in videos:
        print(f"\n{'=' * 40}\n🎥 Processing: {video_name}\n{'=' * 40}")

        try:
            my_model.set_video_meta_data(video_name, args.VidQry)
        except Exception as e:
            print(f"❌ Error setting video metadata: {e}")
            continue

        # 检查所有 PredMetaData 是否已存在
        pred_files_exist = True
        for W in range(1, 5):
            output_dir = os.path.join(args.work_dir, f"PredMetaData_{W}")
            output_file = os.path.join(output_dir, f"{video_name}.json")
            if not os.path.exists(output_file):
                pred_files_exist = False
                break
        if pred_files_exist:
            print(f"⏩ Results exist for all PredMetaData directories, skipping: {video_name}")
            continue

        torch.cuda.empty_cache()

        # 1. 场景检测
        try:
            print("🔍 Detecting scenes...")
            scene_list, start_frames = my_model.detect_scences(None)
            if start_frames[-1] < my_model.n_frames:
                start_frames.append(my_model.n_frames)
            print(f"✅ Detected {len(start_frames) - 1} scenes")
        except Exception as e:
            print(f"❌ Scene detection failed: {e}")
            continue

        # 2. 特征处理
        try:
            print("🖼️ Fetching frames...")
            frames = fetch_frames(my_model.video_path)

            frame_emb_file = os.path.join(frame_emb_dir, f"{video_name}")
            print(f"🔧 Frame embedding file path: {frame_emb_file}")

            frame_emb_exists = False
            possible_paths = [f"{frame_emb_file}.npy", frame_emb_file]
            for path in possible_paths:
                if os.path.exists(path):
                    frame_emb_file = path
                    frame_emb_exists = True
                    print(f"✅ Found existing frame embeddings at: {path}")
                    break

            if not frame_emb_exists:
                print("📊 Generating frame embeddings...")
                embedder.cache_frame_embeddings(frame_emb_file, frames)
                print(f"✅ Generated frame embeddings at: {frame_emb_file}")
            else:
                print("⏩ Using cached frame embeddings")
        except Exception as e:
            print(f"❌ Frame processing failed: {e}")
            continue

        # 3. 场景合并
        try:
            if os.path.exists(frame_emb_file):
                print("🔀 Merging scenes...")
                start_frames = my_model.merge_scenes(start_frames, frame_emb_file, min_frames=150)
                if start_frames[-1] < my_model.n_frames:
                    start_frames.append(my_model.n_frames)
                print(f"✅ Merged into {len(start_frames) - 1} scenes")
            else:
                print(f"⚠️ Frame embedding file not found: {frame_emb_file}")
        except Exception as e:
            print(f"⚠️ Scene merging failed: {e}, using original scenes")

        # 4. 生成描述
        scene_desc_file = os.path.join(scene_desc_dir, f"{video_name}.json")
        try:
            if not os.path.exists(scene_desc_file):
                print("📝 Generating scene descriptions...")
                my_model.generate_scene_descriptions(frames, start_frames, output_file=scene_desc_file)
                print(f"✅ Scene descriptions saved to: {scene_desc_file}")
            else:
                print("⏩ Using existing scene descriptions")
        except Exception as e:
            print(f"❌ Scene description generation failed: {e}")
            continue

        # 5. 计算评分（tvsum_v2 target-only + optional context）
        try:
            print("🧮 Computing scene scores (tvsum_v2)...")
            scene_scores = scorer.compute_scenes_score(scene_desc_file, args.user_query)
            if scene_scores is None:
                raise RuntimeError("Scene scoring returned no results")

            required_scores = len(start_frames) - 1
            if len(scene_scores) != required_scores:
                print(f"⚠️ Scene scores count mismatch: {len(scene_scores)} vs {required_scores}")
                if len(scene_scores) > required_scores:
                    scene_scores = scene_scores[:required_scores]
                else:
                    # 回填最后一个分数
                    scene_scores = scene_scores + [scene_scores[-1] if scene_scores else 50] * (required_scores - len(scene_scores))
        except Exception as e:
            print(f"❌ Scene scoring failed: {e}")
            continue

        # 6. 保存结果 - 对每个窗口大小分别处理（保持你原有的后续流程）
        try:
            for W in range(1, 5):
                output_dir = os.path.join(args.work_dir, f"PredMetaData_{W}")
                output_file = os.path.join(output_dir, f"{video_name}.json")

                if not os.path.exists(output_file):
                    print(f"💾 Saving results for PredMetaData_{W}...")
                    os.makedirs(output_dir, exist_ok=True)

                    my_model.window_size = W
                    my_model.prediciton_meta_data_dir = output_dir
                    my_model.set_video_meta_data(video_name, args.user_query)

                    consistency, dissimilarity = my_model.calc_frames_data(
                        cluster_algo, start_frames, len(frames)
                    )

                    my_model.prediction_meta_data_file = output_file

                    my_model.save_results(
                        scene_scores,
                        start_frames,
                        consistency,
                        dissimilarity,
                        args.user_query
                    )
                    print(f"✅ Saved results to: {output_file}")
                else:
                    print(f"⏩ Results for PredMetaData_{W} already exist, skipping")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            continue

        print(f"✅ {video_name} DONE!")

    print("\n✅ All videos processed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Scene Scoring Pipeline (tvsum_v2 target-only + context, GPT-5)")
    parser.add_argument("--video_name", type=str, default="", help="Specific video to process")
    parser.add_argument("--video_type", type=str, choices=['mp4', 'webm'], default="mp4", help="Video file format")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--user_query", type=str, default="", help="User query for content preference")
    parser.add_argument("--VidQry", type=str, default="", help="Video query metadata")
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Working directory for output (e.g., /usr1/home/.../tvsum_metadata_gpt)")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output directory")
    parser.add_argument("--description_model_name", type=str,
                        default="lmms-lab/LLaVA-Video-7B-Qwen2", help="Model for video description")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--min_scene_duration", type=int, default=2, help="Minimum scene duration in seconds")
    parser.add_argument("--segment_duration", type=int, default=1, help="Segment duration for processing")
    parser.add_argument("--openai_key", type=str, required=True, help="OpenAI API key")

    args = parser.parse_args()

    # 参数校验
    if not os.path.exists(args.video_dir):
        print(f"❌ Video directory does not exist: {args.video_dir}")
        sys.exit(1)

    if not args.openai_key:
        print("❌ OpenAI API key is required")
        sys.exit(1)

    if not os.path.exists(args.work_dir):
        print(f"⚠️ Work directory does not exist, creating: {args.work_dir}")
        os.makedirs(args.work_dir, exist_ok=True)

    try:
        solve(args)
    except KeyboardInterrupt:
        print("\n🚫 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
