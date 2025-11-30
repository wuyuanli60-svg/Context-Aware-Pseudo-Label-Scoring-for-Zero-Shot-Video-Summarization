# 🎬 Context-Aware Pseudo-Label Scoring for Zero-Shot
Video Summarization

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Model](https://img.shields.io/badge/LLM-GPT--4o-orange.svg)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

This repository implements a **rubric-guided zero-shot video summarization framework**.  
It supports experiments on **TVSum**, **SumMe**, and **QFVS**, including the complete pipeline:

- **Reason Generation** 
- **Core Scoring Module** 
- **Evaluation** 

---

## 📑 Table of Contents

- [Features](#✨ Features)
- [Project Structure](#📂 Project Structure)
- [Environment Setup](#🛠 Environment Setup)
- [Dataset Preparation](#📁 Dataset Preparation)
  - [TVSum](#tvsum)
  - [SumMe](#summe)
  - [QFVS](#qfvs)
- [Pipeline Overview](#pipeline-overview)
  - [1. Reason Generation](#1️⃣-reason-generation)
  - [2. Core Scoring Module](#2️⃣-core-scoring-module)
  - [3. Evaluation](#3️⃣-evaluation)
- [Full End-to-End Example](#full-end-to-end-example)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## ✨ Features

- 🚫 **Zero-shot pipeline** (no supervised training)
- 📊 Evaluated on **TVSum / SumMe / QFVS**
- 🧠 Rubric-guided LLM scoring with contextual reasoning
- 📈 Supports frame-level & segment-level outputs
- ⚙ Config-based pipeline, easy to reproduce
- 🔍 Clean modular design (Reason → Scoring → Evaluation)

---

## 📂 Project Structure

├── Data_Evaluation/

│   ├── evaluation/
│   ├── scripts/
│   └── splits/
│   ├── sumMe_mapping.json

│   └── tvSum_mapping.json

├── Generate_Reason/

│   ├── QFVS-reason-generation.py
│   └── TVSum_and_Summe_Generation.py
├── QFVS/
│   ├── evaluation/
│   ├── GFVS_splits.py

│   ├── QFVS_mapping.py

│   ├── QFVS_mapping.json

│   └── QFVS_splits.json

├── src/
│   ├──model/
│   └── utils.py
├── requirements.txt
└── README.md

## 🛠 Environment Setup

Clone and install:

```bash
git clone https://github.com/<yourname>/<your-project>.git
cd <your-project>
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY"
```

## 📁 Dataset Preparation

## Dataset Preparation

### TVSum  
Download the official TVSum dataset here:  
https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz  

Note: This archive contains the 50 videos, along with shot-level annotations for importance and metadata.  

### SumMe  
Download the SumMe dataset here:  
https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip  

Note: This includes the original videos and ground truth summary annotations.  

### QFVS (Query-Focused Video Summarization)  
A publicly-available version of QFVS is hosted on Hugging Face (ToughStone / QFVS collection):  
https://huggingface.co/ToughStone/QFVS  

You can load the data via Hugging Face datasets API or download the files directly.  

## 🔄 Pipeline Overview

### 1️⃣ **Reason Generation**

------

This part of the reason-generation process requires **pre-computed segment scores** before the explanations can be produced.
 The chain-of-thought (CoT) reasoning is generated using **GPT-5**, and the resulting explanations have already been integrated into the downstream scoring pipeline.

Therefore, this script is provided **for documentation and demonstration purposes only** and does **not** affect the subsequent scoring process.

### TVSUM/SUMME Reason Generation

```bash
python Generate_Reason/generate_reasons.py \
  --scores_all outputs/scene_scores_results.json \
  --descs_dir data/descriptions/ \
  --out_dir outputs/reasons/per_video/ \
  --batch_out outputs/reasons/reasons_all.json \
  --model gpt-5 \
  --limit 10 \
  --k_high 3 \
  --k_low 3
```

### QFVS Reason Generation

```bash
python Generate_Reason/rank_segments_by_gt_points.py \
  --gt data/tvsum/gt_points/video01_points.txt \
  --segments_json data/tvsum/segments/video01_segments.json \
  --descs_json data/tvsum/descriptions/video01_descs.json \
  --k_high 3 \
  --k_low 3 \
  --model gpt-5 \
  --out outputs/reasons_gt/video01_reasons.json
```

### 2️⃣ **Core Scoring Module**

### TVSum Scoring Gneration



### SumMe Scoring Gneration



### QFVS Scoring Gneration



### 3️⃣ **Evaluation**

### TVSum\SumMe Evaluation



### QFVS Evaluation



## 📊 Results



## 📚 Citation

@article{your2025paper,
  title={Rubric-Guided Zero-Shot Video Summarization},
  author={Your Name},
  journal={ArXiv},
  year={2025}
}

## 📄 License

This project is licensed under the **MIT License**.