import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
import sys

import numpy as np
from evaluator import evaluate_summaries
from src.model.heuristic_prediction import heuristic_predicator


def run(args):
    work_dir = args.work_dir
    video_name = args.video_name
    gt_file = args.gt_file
    splits_file = args.splits_file
    mapping_file = args.mapping_file
    metric = args.metric
    meta_data_dir = args.meta_data_dir

    # ========== 新增调试信息 ==========
    print("\n" + "=" * 50)
    print("DEBUG: Input Arguments")
    print(f"work_dir: {work_dir}")
    print(f"meta_data_dir: {meta_data_dir}")
    print(f"norm: {args.norm}")
    print("=" * 50 + "\n")
    # ================================

    splites = None
    with open(splits_file, 'r') as json_file:
        splites = json.load(json_file)

    with open(mapping_file, 'r') as json_file:
        mapping = json.load(json_file)

    # ========== 修改1：更安全的文件列表获取方式 ==========
    meta_data_files = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(meta_data_dir, 'PredMetaData_1'))
                       if f.endswith('.json')]
    print(f"DEBUG: Found {len(meta_data_files)} meta data files")
    print(f"Sample files: {meta_data_files[:3]}...\n")
    # =============================================

    data = {}
    missing_files = []

    for video_name in meta_data_files:
        data[video_name] = {}
        for i in range(1, 5):
            file_path = os.path.join(meta_data_dir, f'PredMetaData_{i}', f'{video_name}.json')
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    # ========== 修改2：验证必要字段存在 ==========
                    required_fields = ['n_frames', 'scene_scores', 'scene_frames']
                    if not all(field in content for field in required_fields):
                        print(f"WARNING: Missing fields in {file_path}")
                        missing_files.append(file_path)
                        continue
                    data[video_name][i] = content
                    # 调试打印前3个文件
                    if len(data) <= 3 and i == 1:
                        print(f"DEBUG: Loaded {video_name}")
                        print(f"  n_frames: {content['n_frames']}")
                        print(f"  scores: {content['scene_scores'][:3]}...\n")
            except Exception as e:
                print(f"ERROR loading {file_path}: {str(e)}")
                missing_files.append(file_path)

    # ========== 修改3：检查缺失文件 ==========
    if missing_files:
        print(f"\n❌ Missing or invalid files ({len(missing_files)}):")
        for f in missing_files[:3]:  # 只打印前3个示例
            print(f" - {f}")
        if len(missing_files) > 3:
            print(f" ... and {len(missing_files) - 3} more")
    # =====================================

    Norm = args.norm
    results_all = []

    for i, split in enumerate(splites):
        print(f'\n{"=" * 30} Split {i + 1} {"=" * 30}')
        test_keys = split['test_keys']
        test_videos = [mapping[test_k] for test_k in test_keys]

        # ========== 修改4：验证测试视频数据存在 ==========
        missing_test_videos = [v for v in test_videos if v not in data]
        if missing_test_videos:
            print(f"❌ Missing data for test videos: {missing_test_videos[:3]}...")
            continue
        # ==========================================

        test_data = {key: data[key] for key in test_videos}
        output_file_test = os.path.join(work_dir, 'eval.json')

        # ========== 调试信息：打印测试数据样本 ==========
        sample_video = list(test_data.keys())[0]
        print(f"\nDEBUG: Sample test data for {sample_video}:")
        print(f"n_frames: {test_data[sample_video][1]['n_frames']}")
        print(f"scores: {test_data[sample_video][1]['scene_scores'][:5]}...")
        # ==========================================

        heuristic_predicator(test_data, output_file_test, norm=Norm)

        # ========== 调试信息：检查预测文件 ==========
        if os.path.exists(output_file_test):
            with open(output_file_test, 'r') as f:
                pred_content = json.load(f)
                print(f"\nDEBUG: Predictions for {sample_video}:")
                print(json.dumps(pred_content.get(sample_video, {}), indent=2)[:200] + "...")
        # =====================================

        split_f1, split_p, split_r = evaluate_summaries(
            output_file_test, gt_file, test_keys, mapping, metric, test=True
        )
        results_all.append((split_f1, split_p, split_r))
        print(f"Current Split - F1: {split_f1:.2f}  |  Precision: {split_p:.2f}  |  Recall: {split_r:.2f}")

    # clean up
    if os.path.exists(work_dir + f'/eval.json'):
        os.remove(work_dir + f'/eval.json')

    print(f'\n{"=" * 50}')
    print(f'Final Results for {metric} dataset:')
    if results_all:
        mean_f1 = float(np.mean([r[0] for r in results_all]))
        mean_p = float(np.mean([r[1] for r in results_all]))
        mean_r = float(np.mean([r[2] for r in results_all]))
        print(f'Precision : {mean_p:.2f}%')
        print(f'Recall    : {mean_r:.2f}%')
        print(f'F1-score  : {mean_f1:.2f}%')
    else:
        print('No valid splits evaluated.')
    print("=" * 50 + '\n')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper Param Search')
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--video_name", type=str, default="")
    parser.add_argument("--gt_file", type=str, help='Ground truth file path')
    parser.add_argument("--splits_file", type=str)
    parser.add_argument("--mapping_file", type=str, help="videos directory")
    parser.add_argument("--meta_data_dir", type=str, help="videos directory")
    parser.add_argument("--metric", type=str, choices=['summe', 'tvsum'])
    parser.add_argument("--norm", type=str, choices=['None', 'MinMax', 'Exp', 'MinMax+Exp'])
    args = parser.parse_args()
    run(args)