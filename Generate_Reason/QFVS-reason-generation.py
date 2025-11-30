#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, time, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

# ============ Utils ============

def _to_int(x) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None

def _norm_int_list(lst) -> List[int]:
    out=[]
    for x in lst:
        v=_to_int(x)
        if v is not None: out.append(v)
    return out

def interval_length(iv: Tuple[int,int]) -> int:
    s,e = iv
    return max(0, e - s + 1)

def merge_intervals(iv: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not iv: return []
    iv = sorted(iv, key=lambda x: (x[0], x[1]))
    out=[iv[0]]
    for s,e in iv[1:]:
        ps,pe = out[-1]
        if s <= pe + 1:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s,e))
    return out

def boundaries_to_segments(bounds: List[int]) -> List[Tuple[int,int]]:
    """把单调非降的边界列表 [b0,b1,...,bn] 转闭区间 [(b0,b1-1),...,(b_{n-1},b_n-1)]"""
    b = _norm_int_list(bounds)
    segs=[]
    for i in range(len(b)-1):
        s, e = b[i], b[i+1]-1
        if e < s: s,e = e,s
        segs.append((s,e))
    return segs

def parse_pair_list(lst) -> List[Tuple[int,int]]:
    """解析 [[s,e],...] 或 [{'start':..,'end':..},...]"""
    segs=[]
    for it in lst:
        s=e=None
        if isinstance(it, list) and len(it)>=2:
            s,e=_to_int(it[0]), _to_int(it[1])
        elif isinstance(it, dict):
            for ks,ke in (("start","end"), ("start_frame","end_frame"), ("s","e"),
                          ("beg","end"), ("from","to"), ("left","right")):
                if s is None and ks in it: s=_to_int(it.get(ks))
                if e is None and ke in it: e=_to_int(it.get(ke))
        if s is None or e is None: continue
        if e < s: s,e=e,s
        segs.append((s,e))
    return segs

# ============ GT Loader（点集合！） ============

def load_gt_indices(path: Path) -> List[int]:
    """
    把 oracle.txt / json 解析为“已排序去重”的索引集合（整数列表），
    每个值代表一个“重要帧/shot 索引”，不扩成区间。
    支持：
      - 纯文本：每行一个整数；若某行是 "s e" 两个数，取 s 和 e 两个点（兼容旧习）
      - JSON：{"gt_idx"/"gt"/"indices"/"frames":[...]} 或直接 [idx...]
              若 {"gtsummary":[0/1,...]} 则取所有 1 的索引
    """
    text = path.read_text(encoding="utf-8").strip()
    pts: List[int] = []

    # JSON
    if text[:1] in ("{","["):
        obj = json.loads(text)
        if isinstance(obj, dict):
            if "gtsummary" in obj and isinstance(obj["gtsummary"], list):
                for i, v in enumerate(obj["gtsummary"]):
                    if str(v).strip() in ("1","True","true") or v==1:
                        pts.append(i)
            else:
                for key in ("gt_idx","gt","indices","frames"):
                    if key in obj and isinstance(obj[key], list):
                        pts.extend(_norm_int_list(obj[key]))
                        break
        elif isinstance(obj, list):
            pts.extend(_norm_int_list(obj))
    else:
        # 纯文本：一列（常见），或两列（兼容旧文件：分别当作两个点）
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            sp = ln.split()
            if len(sp) == 1:
                v = _to_int(sp[0])
                if v is not None: pts.append(v)
            else:
                v1 = _to_int(sp[0]); v2 = _to_int(sp[1])
                if v1 is not None: pts.append(v1)
                if v2 is not None: pts.append(v2)

    # 去重升序
    pts = sorted(set([p for p in pts if p is not None]))
    if not pts:
        raise ValueError(f"Empty GT indices after parsing: {path}")
    return pts

# ============ Segments Loader（加上 start_frames 支持） ============

def load_segments_scene_frames(path: Path) -> List[Tuple[int,int]]:
    """
    读取片段边界/区间：
    优先级：
      1) scene_frames
         - 若元素是数字：当作边界列表 → 邻接成段
         - 若元素是 [s,e] 或 {start,end}：直接用
      2) start_frames（常见于 PredMetaData_*）：边界列表 → 邻接成段
      3) 其它常见键：segments / scenes / scene_list / shots（对 [s,e] / {start,end}）
      4) 根是列表：同上
      5) 宽松：在 dict 中找“像列表”的字段
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    def _try_as_boundaries(node):
        if isinstance(node, list) and node and isinstance(node[0], (int, float, str)):
            ints = _norm_int_list(node)
            if len(ints) >= 2:  # 至少能成 1 段
                return boundaries_to_segments(ints)
        return []

    if isinstance(obj, dict):
        # 1) scene_frames
        if isinstance(obj.get("scene_frames"), list):
            # 1.1 尝试边界
            segs = _try_as_boundaries(obj["scene_frames"])
            if segs: return segs
            # 1.2 尝试 [s,e] / {start,end}
            segs = parse_pair_list(obj["scene_frames"])
            if segs: return segs

        # 2) start_frames
        if isinstance(obj.get("start_frames"), list):
            segs = _try_as_boundaries(obj["start_frames"])
            if segs: return segs

        # 3) 其它常见键（区间列表）
        for key in ("segments","scenes","scene_list","shots"):
            if isinstance(obj.get(key), list):
                segs = parse_pair_list(obj[key])
                if segs: return segs

        # 4) 宽松：任意字段长得像“列表的列表/字典”
        for _, node in obj.items():
            if isinstance(node, list):
                segs = parse_pair_list(node)
                if segs: return segs
                segs = _try_as_boundaries(node)
                if segs: return segs

    # 5) 根就是列表
    if isinstance(obj, list):
        segs = parse_pair_list(obj)
        if segs: return segs
        segs = boundaries_to_segments(obj)
        if segs: return segs

    raise ValueError(f"No valid segments found in: {path}  (expected scene_frames/start_frames/segments/scenes/scene_list/shots)")

# ============ Descriptions Loader (兼容 scene_N_description) ============

_SCENE_DESC_RE = re.compile(r"^scene_(\d+)_description$", re.IGNORECASE)

def load_descs_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    video_desc = ""
    seg_descs: List[str] = []

    if isinstance(obj, dict):
        video_desc = str(obj.get("video_description", "")) or str(
            obj.get("video_desc", obj.get("overview", obj.get("title", obj.get("summary", ""))))
        )
        numbered=[]
        for k,v in obj.items():
            m = _SCENE_DESC_RE.match(str(k))
            if m and isinstance(v, str):
                numbered.append((int(m.group(1)), v))
        if numbered:
            numbered.sort(key=lambda x: x[0])
            seg_descs = [s for _,s in numbered]

        if not seg_descs:
            for key in ("segments","shots","scenes","captions","sentences","data","items","descs","descriptions"):
                node = obj.get(key)
                if isinstance(node, list):
                    tmp=[]
                    for it in node:
                        if isinstance(it, str): tmp.append(it)
                        elif isinstance(it, dict):
                            for dk in ("desc","text","caption","sentence","summary"):
                                if dk in it and isinstance(it[dk], str):
                                    tmp.append(it[dk]); break
                    if tmp:
                        seg_descs = tmp
                        break
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], str):
            seg_descs = [str(x) for x in obj]
        elif obj and isinstance(obj[0], dict):
            tmp=[]
            for it in obj:
                for dk in ("desc","text","caption","sentence","summary"):
                    if dk in it and isinstance(it[dk], str):
                        tmp.append(it[dk]); break
            seg_descs = tmp

    if not seg_descs:
        raise ValueError(f"No segment descriptions found in {path}")
    return {"video_desc": video_desc, "segments": seg_descs}

# ============ Scoring by GT “点命中计数” ============

def hit_counts_scores(gt_points: List[int], segs: List[Tuple[int,int]]) -> List[int]:
    """每段统计：有多少 GT 索引 p 落在 [s,e] 内"""
    pts = sorted(gt_points)
    scores=[]
    j=0
    for (s,e) in segs:
        cnt=0
        # 前移到 >= s
        while j < len(pts) and pts[j] < s:
            j += 1
        k = j
        while k < len(pts) and pts[k] <= e:
            cnt += 1
            k += 1
        scores.append(cnt)
    return scores

def pick_top_low(descs: List[str], segs: List[Tuple[int,int]], scores: List[int],
                 k_high=3, k_low=3, char_limit: int=420) -> Dict[str, Any]:
    n = min(len(descs), len(segs), len(scores))
    if n == 0:
        raise ValueError("Nothing to rank: empty descs/segs/scores.")
    triples = [(i, descs[i], segs[i], int(scores[i])) for i in range(n)]
    top = sorted(triples, key=lambda x: (-x[3], x[0]))[:k_high]
    low = sorted(triples, key=lambda x: (x[3], x[0]))[:k_low]

    def clean_text(s: str) -> str:
        s = re.sub(r"\s+"," ", (s or "")).strip()
        return s[:char_limit] + " ..." if len(s)>char_limit else s

    return {
        "scores": [x[3] for x in triples],  # 命中个数
        "top": [{"idx":x[0], "seg":x[2], "score":x[3], "desc":clean_text(x[1])} for x in top],
        "low": [{"idx":x[0], "seg":x[2], "score":x[3], "desc":clean_text(x[1])} for x in low],
    }

# ============ GPT Prompt & Call（GPT-5 极简参数 + 超时重试） ============

def build_prompt(video_id, video_desc, top_list, low_list):
    def extract_descs(lst): return [it["desc"] for it in lst]
    high_snips = extract_descs(top_list)
    low_snips  = extract_descs(low_list)

    def toks(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", text or "")
    def top_words(snips: List[str], top_n=20):
        freq={}
        for s in snips:
            for t in toks(s):
                tl=t.lower()
                if len(tl)<3 or tl in {"the","and","for","you","are","was","were","with","from","into","this","that","there","here","she","he","they","we","but","not","can"}:
                    continue
                freq[tl]=freq.get(tl,0)+1
        return [w for w,_ in sorted(freq.items(), key=lambda kv:(-kv[1],kv[0]))][:top_n]

    hints_hi = top_words(high_snips)
    hints_lo = top_words(low_snips)

    return f"""
You will write THREE concrete reasons for this video and return STRICT JSON with the keys:
- "reason_positive": one succinct but specific reason why the HIGH-score segments are key.
- "reason_negative": one succinct but specific reason why the LOW-score segments are not key.
- "reason_difference": one succinct but specific reason explaining their essential difference.

Writing requirements:
- Ground every reason in the provided descriptions (do not invent entities).
- Prefer concrete facts: WHO did WHAT action → WHAT outcome/visible change (before → after), and HOW this ties to the overall goal below.
- You may summarize across segments; do not list them one by one; do not include indices.
- 2–4 sentences per reason; avoid boilerplate.

[VIDEO_ID]
{video_id}

[GLOBAL DESCRIPTION]
{video_desc or ""}

[HIGH-SCORE SEGMENTS]  # reference only
{json.dumps(high_snips, ensure_ascii=False, indent=2)}

[LOW-SCORE SEGMENTS]   # reference only
{json.dumps(low_snips, ensure_ascii=False, indent=2)}

[OPTIONAL CONTENT HINTS — HIGH]
{json.dumps(hints_hi, ensure_ascii=False, indent=2)}

[OPTIONAL CONTENT HINTS — LOW]
{json.dumps(hints_lo, ensure_ascii=False, indent=2)}

Return ONLY JSON with those three keys. No extra text.
""".strip()

def call_gpt_json(prompt: str, model: str="gpt-5",
                  max_tokens: int=800, temperature: Optional[float]=0.3) -> str:
    """
    极简/稳妥版本（贴合你原来方式）：
    - 如果是 GPT-5：只传 model+messages（不传 max_tokens/response_format/temperature）
    - 如果不是 GPT-5：按你原来风格传 temperature 和 max_tokens
    - 加 request_timeout 和有限重试，避免“卡住”
    """
    import time
    try:
        import openai
    except Exception:
        raise RuntimeError("`openai` package not found. Please `pip install openai==0.28.1`")

    if not hasattr(openai, "ChatCompletion"):
        raise RuntimeError("Your installed `openai` SDK is incompatible. Please install openai<=0.28.1")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key

    is_gpt5 = str(model).lower().startswith("gpt-5")
    MAX_TRIES = 4
    TIMEOUT = 60  # seconds
    SLEEP_BASE = 2.0

    def _once_gpt5():
        return openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with a single valid JSON object and nothing else."},
                {"role": "user", "content": prompt},
            ],
            request_timeout=TIMEOUT,
        )["choices"][0]["message"]["content"]

    def _once_legacy():
        return openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with a single valid JSON object and nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=(temperature if temperature is not None else 1),
            max_tokens=max_tokens,
            request_timeout=TIMEOUT,
        )["choices"][0]["message"]["content"]

    tries = 0
    last_err = None
    while tries < MAX_TRIES:
        tries += 1
        try:
            if is_gpt5:
                return _once_gpt5()
            else:
                return _once_legacy()
        except Exception as e:
            msg = str(e)
            last_err = e
            if ("429" in msg or "rate limit" in msg.lower() or "timeout" in msg.lower()):
                time.sleep(min(60, SLEEP_BASE ** tries)); continue
            if ("Only the default (1) value is supported" in msg and "temperature" in msg):
                try:
                    return openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Reply with a single valid JSON object and nothing else."},
                            {"role": "user", "content": prompt},
                        ],
                        request_timeout=TIMEOUT,
                    )["choices"][0]["message"]["content"]
                except Exception as e2:
                    last_err = e2
                    time.sleep(min(60, SLEEP_BASE ** tries)); continue
            time.sleep(min(60, SLEEP_BASE ** tries))
    raise last_err

def _value_to_text(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, str): return v
    if isinstance(v, (int, float, bool)): return str(v)
    if isinstance(v, list):
        parts = [_value_to_text(x) for x in v]
        return ". ".join([p.strip().rstrip(".") for p in parts if p.strip()]).strip()
    if isinstance(v, dict):
        parts=[]
        for k in sorted(v.keys()):
            parts.append(_value_to_text(v[k]))
        return ". ".join([p.strip().rstrip(".") for p in parts if p.strip()]).strip()
    return str(v)

def _deep_get_first(obj: Any, keys: List[str]) -> Optional[Any]:
    if isinstance(obj, dict):
        for k in keys:
            for kk in obj.keys():
                if kk.lower().replace("-", "_") == k.lower().replace("-", "_"):
                    return obj[kk]
        for v in obj.values():
            got = _deep_get_first(v, keys)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for it in obj:
            got = _deep_get_first(it, keys)
            if got is not None:
                return got
    return None

POS_ALIASES = ["reason_positive","positive_reason","positive","pos","reasons_positive"]
NEG_ALIASES = ["reason_negative","negative_reason","negative","neg","reasons_negative"]
DIF_ALIASES = ["reason_difference","difference_reason","difference","diff","reasons_difference"]

def safe_json_parse(text: str) -> Union[dict, list]:
    try:
        return json.loads(text)
    except Exception:
        s = text.find("{"); e = text.rfind("}")
        if s!=-1 and e!=-1 and e>s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                pass
        try:
            return json.loads(text.strip(" \n\r\t"))
        except Exception:
            return {}

def finalize_reasons(obj: Union[dict, list]) -> dict:
    if not isinstance(obj, (dict, list)): obj = {}
    raw_pos = _deep_get_first(obj, POS_ALIASES)
    raw_neg = _deep_get_first(obj, NEG_ALIASES)
    raw_dif = _deep_get_first(obj, DIF_ALIASES)
    pos_txt = _value_to_text(raw_pos).strip()
    neg_txt = _value_to_text(raw_neg).strip()
    dif_txt = _value_to_text(raw_dif).strip()
    if not pos_txt: pos_txt = "No reason generated."
    if not neg_txt: neg_txt = "No reason generated."
    if not dif_txt: dif_txt = "No reason generated."
    return {"reason_positive": pos_txt, "reason_negative": neg_txt, "reason_difference": dif_txt}

# ============ Main ============

def main():
    ap = argparse.ArgumentParser("Rank segments by GT *point* hits; pick Top/Low; ask GPT for three reasons.")
    ap.add_argument("--gt", required=True, help="GT file: txt or json (points/indices)")
    ap.add_argument("--segments_json", required=True, help="JSON with scene_frames/start_frames or similar")
    ap.add_argument("--descs_json", required=True, help="JSON with descriptions (scene_N_description, etc.)")
    ap.add_argument("--k_high", type=int, default=3)
    ap.add_argument("--k_low",  type=int, default=3)
    ap.add_argument("--model",  default="gpt-5")
    ap.add_argument("--out",    required=True, help="If directory: write <desc_stem>.reasons.json; if file: write to file")
    args = ap.parse_args()

    # 1) 读取 GT（点集合）
    gt_points = load_gt_indices(Path(args.gt))

    # 2) 读取段（支持 start_frames 边界列表）
    segs    = load_segments_scene_frames(Path(args.segments_json))

    # 3) 读取描述
    descobj = load_descs_json(Path(args.descs_json))
    descs   = descobj["segments"]
    video_desc = descobj.get("video_desc","")
    vid     = Path(args.descs_json).stem

    # 4) 对齐长度
    n0, n1 = len(segs), len(descs)
    n = min(n0, n1)
    if n == 0:
        raise ValueError(f"Empty after alignment (segments={n0}, descs={n1})")
    if n0 != n1:
        print(f"[warn] segments({n0}) != descs({n1}); trunc to {n}")
        segs, descs = segs[:n], descs[:n]

    # 5) 命中计数打分
    scores = hit_counts_scores(gt_points, segs)
    picked = pick_top_low(descs, segs, scores, k_high=args.k_high, k_low=args.k_low)

    # 6) GPT 理由
    prompt = build_prompt(vid, video_desc, picked["top"], picked["low"])
    raw = call_gpt_json(prompt, model=args.model, temperature=0.3, max_tokens=800)
    reasons = finalize_reasons(safe_json_parse(raw))

    # 7) 仅保存理由
    out = {
        "reason_positive": reasons["reason_positive"],
        "reason_negative": reasons["reason_negative"],
        "reason_difference": reasons["reason_difference"],
    }

    out_path = Path(args.out)
    if out_path.exists() and out_path.is_dir():
        out_file = out_path / f"{vid}.reasons.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path

    out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 终端简要输出（命中计数）
    print("\n== Top (by GT-hit count) ==")
    for i, it in enumerate(picked["top"], 1):
        print(f"[{i}] idx={it['idx']} seg={tuple(it['seg'])} hits={it['score']} :: {it['desc']}")
    print("\n== Low (by GT-hit count) ==")
    for i, it in enumerate(picked["low"], 1):
        print(f"[{i}] idx={it['idx']} seg={tuple(it['seg'])} hits={it['score']} :: {it['desc']}")

    print("\n== GPT Reasons ==")
    print("POS:", out["reason_positive"])
    print("NEG:", out["reason_negative"])
    print("DIF:", out["reason_difference"])
    print(f"\n[OK] written -> {out_file}")

if __name__ == "__main__":
    main()
