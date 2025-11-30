#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch reasons (NO final summary).
- Read one scores JSON for ALL videos.
- Read the first N description JSONs under --descs_dir.
- For each video:
  1) Align segment descriptions with scores.
  2) Pick Top-K (HIGH) and Low-K (LOW) segments.
  3) Ask an LLM for THREE concrete reasons:
        { "reason_positive", "reason_negative", "reason_difference" }
  4) Save per-video JSON: out_dir/<VIDEO_ID>.reasons.json
- Also write a combined JSON (keys are video IDs only; NO _summary) to --batch_out.

Example:
python test.py \
  --scores_all "/path/scene_scores_results.json" \
  --descs_dir  "/path/tvsum_metadata/sceneDesc" \
  --out_dir    "/path/reasons_out" \
  --batch_out  "/path/top10_reasons.json" \
  --model      "gpt-3.5-turbo" \
  --limit      10 --k_high 3 --k_low 3
"""

import os, re, json, time, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# =========================
# Loaders
# =========================

def load_scores_all(path: Path, score_field: str = None) -> Dict[str, list]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cand_score_keys = (["scene_scores"] +
                       ([score_field] if score_field else []) +
                       ["scores", "gt", "score", "importance", "value", "y", "s"])

    def norm_vid(v): return Path(str(v)).stem
    out: Dict[str, list] = {}

    def coerce_seq(seq):
        arr=[]
        if not isinstance(seq, list): return arr
        for x in seq:
            if isinstance(x, (int, float)):
                arr.append(float(x))
            elif isinstance(x, str):
                try: arr.append(float(x))
                except: pass
            elif isinstance(x, dict):
                for k in cand_score_keys:
                    if k in x:
                        val = x[k]
                        if isinstance(val, (int,float,str)):
                            try:
                                arr.append(float(val)); break
                            except: pass
                        elif isinstance(val, dict) and "value" in val:
                            try:
                                arr.append(float(val["value"])); break
                            except: pass
        return arr

    def try_add(vid, node):
        if vid is None or node is None: return
        if isinstance(node, dict):
            for k in cand_score_keys:
                if k in node and isinstance(node[k], list):
                    arr = coerce_seq(node[k])
                    if arr:
                        out[norm_vid(vid)] = arr
                        return
        if isinstance(node, list):
            arr = coerce_seq(node)
            if arr:
                out[norm_vid(vid)] = arr
                return

    if isinstance(raw, dict):
        for k, v in raw.items(): try_add(k, v)
        for container in ("videos", "results", "data", "items", "payload"):
            if isinstance(raw.get(container), dict):
                for k, v in raw[container].items(): try_add(k, v)
    elif isinstance(raw, list):
        for it in raw:
            if isinstance(it, dict):
                vid = it.get("video_id") or it.get("video") or it.get("id") or it.get("name")
                node = it.get("scores") or it
                try_add(vid, node)

    if not out:
        raise ValueError(f"Unrecognized all-scores JSON schema. (file={path})")
    return out

def resolve_video_key(score_map: Dict[str, list], video_id: str, scores_raw: dict) -> str:
    if video_id in score_map: return video_id
    for k in score_map.keys():
        if Path(k).stem == video_id: return k
    if isinstance(scores_raw, dict):
        for k, node in scores_raw.items():
            if isinstance(node, dict) and "video_name" in node:
                if Path(str(node["video_name"])).stem == video_id:
                    return Path(str(k)).stem
    raise KeyError(f"'{video_id}' not found in scores file. Some keys: {list(score_map.keys())[:10]}")

_SCENE_DESC_RE = re.compile(r"^scene_(\d+)_description$", re.IGNORECASE)

def load_video_desc(desc_path: Path) -> Dict[str, Any]:
    obj = json.loads(desc_path.read_text(encoding="utf-8"))
    video_desc = ""
    seg_descs: List[str] = []

    if isinstance(obj, dict):
        video_desc = str(obj.get("video_description", "")) or str(
            obj.get("video_desc", obj.get("overview", obj.get("title", obj.get("summary", ""))))
        )
        numbered: List[Tuple[int, str]] = []
        for k, v in obj.items():
            m = _SCENE_DESC_RE.match(str(k))
            if m and isinstance(v, str):
                numbered.append((int(m.group(1)), v))
        if numbered:
            numbered.sort(key=lambda x: x[0])
            seg_descs = [s for _, s in numbered]
        if not seg_descs:
            segs = (obj.get("segments") or obj.get("shots") or obj.get("scenes") or
                    obj.get("captions") or obj.get("sentences") or obj.get("data") or obj.get("items"))
            if isinstance(segs, list):
                for s in segs:
                    if isinstance(s, dict):
                        for dk in ("desc","text","caption","sentence","summary"):
                            if dk in s: seg_descs.append(str(s[dk])); break
                    elif isinstance(s, str):
                        seg_descs.append(s)
            elif isinstance(segs, dict):
                for _, val in sorted(segs.items(), key=lambda kv: kv[0]):
                    if isinstance(val, str):
                        seg_descs.append(val)
                    elif isinstance(val, dict):
                        for dk in ("desc","text","caption","sentence","summary"):
                            if dk in val: seg_descs.append(str(val[dk])); break
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], str):
            seg_descs = [str(x) for x in obj]
        elif isinstance(obj[0], dict):
            for s in obj:
                for dk in ("desc","text","caption","sentence","summary"):
                    if dk in s: seg_descs.append(str(s[dk])); break

    if not seg_descs:
        raise ValueError(f"No segment descriptions found in {desc_path}")
    return {"video_desc": video_desc, "segments": seg_descs}

# =========================
# TopK / LowK
# =========================

def select_topk_lowk(descs: List[str], scores: List[float],
                     k_high: int=3, k_low: int=3, char_limit: int=420):
    n = min(len(descs), len(scores))
    pairs = list(zip(descs[:n], scores[:n]))
    if not pairs:
        raise ValueError("Empty pairs after alignment. Check descs/scores length.")
    top = sorted(pairs, key=lambda x: x[1], reverse=True)[:k_high]
    low = sorted(pairs, key=lambda x: x[1])[:k_low]

    def clean(xs):
        out=[]
        for s,_ in xs:
            s = re.sub(r"\s+"," ", (s or "")).strip()
            if len(s) > char_limit: s = s[:char_limit] + " ..."
            if s: out.append(s)
        return out
    return clean(top), clean(low)

# =========================
# Hints (optional)
# =========================

def _tokenize_for_hints(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{1,}", text)
    out=[]
    STOP={"a","an","the","and","or","but","if","while","so","of","to","in","on","for","with",
          "as","by","at","is","are","was","were","it","this","that","these","those","there",
          "here","he","she","they","we","you","i","from","into","about"}
    for t in tokens:
        tl=t.lower()
        if len(tl)<3 or tl in STOP: continue
        out.append(tl)
    return out

def extract_hints(topk: List[str], lowk: List[str], top_n: int = 20) -> Tuple[List[str], List[str]]:
    def top_words(text):
        freq={}
        for tok in _tokenize_for_hints(text):
            freq[tok]=freq.get(tok,0)+1
        return [w for w,_ in sorted(freq.items(), key=lambda kv:(-kv[1],kv[0]))][:top_n]
    return top_words(" ".join(topk)), top_words(" ".join(lowk))

# =========================
# Prompt（轻约束）
# =========================

def build_three_reason_prompt(video_id, video_desc, topk_snips, lowk_snips, high_hints, low_hints):
    hints_block = ""
    if high_hints or low_hints:
        hints_block = f"""
[OPTIONAL CONTENT HINTS — HIGH]
{json.dumps(high_hints, ensure_ascii=False, indent=2)}

[OPTIONAL CONTENT HINTS — LOW]
{json.dumps(low_hints, ensure_ascii=False, indent=2)}
"""
    return f"""
You will write THREE concrete reasons for this video and return STRICT JSON with the keys:
- "reason_positive": one succinct but specific reason why the HIGH-score segments are key.
- "reason_negative": one succinct but specific reason why the LOW-score segments are not key.
- "reason_difference": one succinct but specific reason explaining their essential difference.

Writing requirements (light but important):
- Ground every reason in the provided descriptions (do not invent entities).
- Prefer concrete facts: WHO did WHAT action → WHAT outcome/visible change (before → after), and HOW this ties to the overall goal below.
- You may summarize across segments; do not list them one by one; do not include indices like TOPK[2] or LOWK[1].
- 2–4 sentences per reason; avoid boilerplate (e.g., “engaging/appealing/impactful” without saying how).

[VIDEO_ID]
{video_id}

[GLOBAL DESCRIPTION]
{video_desc or ""}

[HIGH-SCORE SEGMENTS]  # reference only
{json.dumps(topk_snips, ensure_ascii=False, indent=2)}

[LOW-SCORE SEGMENTS]   # reference only
{json.dumps(lowk_snips, ensure_ascii=False, indent=2)}
{hints_block}

Return ONLY JSON with those three keys. No extra text.
""".strip()

# =========================
# OpenAI call（兼容 old/new SDK）
# =========================

def call_gpt_json(prompt: str, model: str="gpt-3.5-turbo",
                  max_tokens: int=1200, temperature: Optional[float]=0.3) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # old SDK <=0.28
    try:
        import openai as openai_legacy
        if hasattr(openai_legacy, "ChatCompletion"):
            openai_legacy.api_key = api_key
            tries = 0
            while True:
                tries += 1
                try:
                    resp = openai_legacy.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role":"system","content":"Reply with a single valid JSON object and nothing else."},
                            {"role":"user","content":prompt},
                        ],
                        temperature=temperature if temperature is not None else 1,
                        max_tokens=max_tokens
                    )
                    return resp["choices"][0]["message"]["content"]
                except Exception as e:
                    msg=str(e)
                    if "429" in msg or "rate limit" in msg.lower():
                        time.sleep(min(60, 2**tries)); continue
                    if "Only the default (1) value is supported" in msg and "temperature" in msg:
                        resp = openai_legacy.ChatCompletion.create(
                            model=model,
                            messages=[
                                {"role":"system","content":"Reply with a single valid JSON object and nothing else."},
                                {"role":"user","content":prompt},
                            ],
                            max_tokens=max_tokens
                        )
                        return resp["choices"][0]["message"]["content"]
                    raise
    except Exception:
        pass

    # new SDK >=1.0
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    tries=0
    while True:
        tries+=1
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role":"system","content":"Reply with a single valid JSON object and nothing else."},
                    {"role":"user","content":prompt},
                ],
                temperature=temperature if temperature is not None else 1,
                max_tokens=max_tokens
            )
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            msg=str(e)
            if "rate limit" in msg.lower() or "429" in msg:
                time.sleep(min(60, 2**tries)); continue
            if "Unsupported parameter" in msg and "max_tokens" in msg:
                kwargs.pop("max_tokens", None)
                kwargs["max_completion_tokens"] = max_tokens
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            if "Only the default (1) value is supported" in msg and "temperature" in msg:
                kwargs.pop("temperature", None)
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            if "response_format" not in kwargs:
                try:
                    kwargs["response_format"] = {"type": "json_object"}
                    resp = client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content
                except Exception:
                    pass
            raise

# =========================
# Parse & finalize（鲁棒）
# =========================

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

def _strip_indices(t: str) -> str:
    if not isinstance(t, str): return ""
    t = re.sub(r'\bTOPK\[\d+\]\b', '', t)
    t = re.sub(r'\bLOWK\[\d+\]\b', '', t)
    t = re.sub(r'\s{2,}', ' ', t).strip()
    return t

def _drop_trailing_fragment(t: str) -> str:
    if not t: return t
    m = re.search(r'(.*?[.!?])\s+([A-Za-z][^.!?]{6,})$', t)
    if m:
        tail = m.group(2)
        if not re.search(r'\b(is|are|was|were|be|being|been|has|have|had|do|does|did|can|could|should|would|may|might|must|present|show|hold|walk|pose|inspect|compare)\b', tail, re.I):
            return m.group(1).strip()
    return t

POS_ALIASES = ["reason_positive","positive_reason","positive","pos","reasons_positive"]
NEG_ALIASES = ["reason_negative","negative_reason","negative","neg","reasons_negative"]
DIF_ALIASES = ["reason_difference","difference_reason","difference","diff","reasons_difference"]

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

def finalize_reasons(obj: Union[dict, list]) -> dict:
    if not isinstance(obj, (dict, list)): obj = {}
    raw_pos = _deep_get_first(obj, POS_ALIASES)
    raw_neg = _deep_get_first(obj, NEG_ALIASES)
    raw_dif = _deep_get_first(obj, DIF_ALIASES)
    pos_txt = _strip_indices(_value_to_text(raw_pos)).strip()
    neg_txt = _strip_indices(_value_to_text(raw_neg)).strip()
    dif_txt = _strip_indices(_value_to_text(raw_dif)).strip()
    pos_txt = _drop_trailing_fragment(pos_txt)
    neg_txt = _drop_trailing_fragment(neg_txt)
    dif_txt = _drop_trailing_fragment(dif_txt)
    if not pos_txt: pos_txt = "No reason generated."
    if not neg_txt: neg_txt = "No reason generated."
    if not dif_txt: dif_txt = "No reason generated."
    return {"reason_positive": pos_txt, "reason_negative": neg_txt, "reason_difference": dif_txt}

# =========================
# Single-video core
# =========================

def generate_reasons_for_video(scores_all_path: Path,
                               desc_path: Path,
                               model: str = "gpt-3.5-turbo",
                               k_high: int = 3, k_low: int = 3,
                               score_field: str = None) -> dict:
    vid = desc_path.stem
    vdesc = load_video_desc(desc_path)
    seg_descs = vdesc["segments"]; video_desc = vdesc.get("video_desc", "")

    scores_raw = json.loads(Path(scores_all_path).read_text(encoding="utf-8"))
    score_map = load_scores_all(scores_all_path, score_field=score_field)
    video_key = resolve_video_key(score_map, vid, scores_raw)
    scores = score_map[video_key]

    topk_snips, lowk_snips = select_topk_lowk(seg_descs, scores, k_high=k_high, k_low=k_low)
    high_hints, low_hints = extract_hints(topk_snips, lowk_snips, top_n=20)

    prompt = build_three_reason_prompt(vid, video_desc, topk_snips, lowk_snips, high_hints, low_hints)
    raw = call_gpt_json(prompt, model=model, temperature=0.3, max_tokens=1200)
    obj = safe_json_parse(raw)
    return finalize_reasons(obj)

# =========================
# Batch runner (NO summary)
# =========================

def run_batch(scores_all: Path, descs_dir: Path, out_dir: Path, batch_out: Path,
              model: str="gpt-3.5-turbo", limit: int=10, k_high: int=3, k_low: int=3,
              score_field: str=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in descs_dir.glob("*.json")])[:limit]
    if not files:
        raise FileNotFoundError(f"No json files under {descs_dir}")

    combined: Dict[str, dict] = {}
    for i, f in enumerate(files, 1):
        vid = f.stem
        try:
            reasons = generate_reasons_for_video(scores_all, f, model=model,
                                                 k_high=k_high, k_low=k_low,
                                                 score_field=score_field)
            (out_dir / f"{vid}.reasons.json").write_text(
                json.dumps(reasons, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[{i}/{len(files)}] saved → {out_dir / (vid+'.reasons.json')}")
            combined[vid] = reasons
        except Exception as e:
            print(f"[{i}/{len(files)}] {vid} FAILED: {e}")
            combined[vid] = {"reason_positive":"No reason generated.",
                             "reason_negative":"No reason generated.",
                             "reason_difference":"No reason generated."}

    batch_out.parent.mkdir(parents=True, exist_ok=True)
    batch_out.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] combined saved → {batch_out}")

# =========================
# CLI（batch only）
# =========================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_all", required=True, help="Path to ALL-scores JSON (scene_scores_results.json)")
    ap.add_argument("--descs_dir",  required=True, help="Dir containing per-video description JSONs")
    ap.add_argument("--out_dir",    required=True, help="Dir to store per-video reasons jsons")
    ap.add_argument("--batch_out",  required=True, help="Output combined json path (no summary)")
    ap.add_argument("--model",      default="gpt-3.5-turbo", help="OpenAI model name (default: gpt-3.5-turbo)")
    ap.add_argument("--limit",      type=int, default=10, help="Number of videos to process (default: 10)")
    ap.add_argument("--k_high",     type=int, default=3)
    ap.add_argument("--k_low",      type=int, default=3)
    ap.add_argument("--score_field", default=None, help="Custom numeric key if not using 'scene_scores'.")
    args = ap.parse_args()

    run_batch(
        scores_all=Path(args.scores_all),
        descs_dir=Path(args.descs_dir),
        out_dir=Path(args.out_dir),
        batch_out=Path(args.batch_out),
        model=args.model,
        limit=args.limit,
        k_high=args.k_high,
        k_low=args.k_low,
        score_field=args.score_field
    )
