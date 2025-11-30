import argparse
import json
import os
import numpy as np
from src.model.heuristic_prediction import heuristic_predicator_v2
from semantic_evaluation import generate_summary_v2, load_videos_tag
from semantic_evaluation import calculate_semantic_matching_all


def fetch_gt_summaris(gt_dir, splits, mapping):
    gt_summaries = {}
    for split in splits:
        test_keys = split['test_keys']
        for key in test_keys:
            gt_file_name = mapping[key]['gt_file']
            video_id = mapping[key]['video_id']
            gt_file = f'{gt_dir}/{video_id}/{gt_file_name}'
            with open(gt_file, 'r') as f:
                gt_summary = [int(line.strip()) for line in f if line.strip()]
                gt_summary = [x - 1 for x in gt_summary]
            gt_summaries[key] = gt_summary
    return gt_summaries


def run(args):
    work_dir = args.work_dir
    splits_file = args.splits_file
    mapping_file = args.mapping_file
    Tags_file = args.Tags_file
    gt_dir = args.gt_dir
    meta_data_dir = args.meta_data_dir  # 包含PredMetaData_1~4的根目录
    Norm = args.norm

    video_shots_tag = load_videos_tag(mat_path=Tags_file)
    metric_dir = f'{work_dir}/Eval_Norm_{Norm}'
    os.makedirs(metric_dir, exist_ok=True)

    # 加载分割和映射文件
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    gt_summaries = fetch_gt_summaris(gt_dir, splits, mapping)

    # 加载数据：按文件夹批量加载（每个文件夹对应一批视频，无需跨文件夹查找）
    data = {}
    # 加载数据：直接存储每个视频的原始数据（无需按文件夹嵌套）
    data = {}
    for folder_id in range(1, 5):  # 处理PredMetaData_1到4
        folder_path = f'{meta_data_dir}/PredMetaData_{folder_id}'
        if not os.path.exists(folder_path):
            print(f"警告：文件夹 {folder_path} 不存在，跳过")
            continue
        # 加载当前文件夹内所有json文件
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                vidQry = file.split('.')[0]  # 视频query ID（如VidQry_1）
                file_path = f'{folder_path}/{file}'
                with open(file_path, 'r') as f:
                    video_data = json.load(f)
                    data[vidQry] = video_data  # 直接存储视频数据，而非嵌套字典

    # 处理每个分割
    splits_f1_score = []
    for split_idx, split in enumerate(splits):
        print(f"Split {split_idx + 1} : {split['test_video_id']}")
        test_keys = split['test_keys']
        # 筛选当前分割的测试数据（仅加载该分割包含的视频）
        test_data = {}
        for key in test_keys:
            if key in data:
                test_data[key] = data[key]
            else:
                print(f"警告：测试数据 {key} 未找到，跳过")

        #print(data[key])
        # 生成预测结果（保留heuristic_predicator）
        output_file_test = f'{work_dir}/_output_file_test_split_{split_idx + 1}.json'
        heuristic_predicator_v2(test_data, output_file_test, norm=Norm)

        # 生成机器摘要并计算F1
        gt_summaries_lens = {k: len(gt_summaries[k]) for k in test_data.keys() if k in gt_summaries}
        machine_summaries = generate_summary_v2(output_file_test, gt_summaries_lens)
        gt_summaries_tmp = {k: gt_summaries[k] for k in machine_summaries.keys() if k in gt_summaries}
        precision, recall, f1 = calculate_semantic_matching_all(
            machine_summaries,
            gt_summaries_tmp,
            video_shots_tag,
            split['test_video_id'] - 1
        )
        splits_f1_score.append(f1)
        print(f"Split {split_idx + 1} F1: {f1:.4f}")

        # 清理临时文件
        if os.path.exists(output_file_test):
            os.remove(output_file_test)

    # 输出结果
    print("\n===== 所有测试集F1分数 =====")
    for i, f1 in enumerate(splits_f1_score):
        print(f"测试集 {i + 1}: {f1:.4f}")
    avg_f1 = np.mean(splits_f1_score)
    print(f"\n平均F1分数: {avg_f1:.4f}")

    # 保存结果
    result_path = f'{metric_dir}/f1_results.json'
    with open(result_path, 'w') as f:
        json.dump({
            "split_f1_scores": {f"split_{i + 1}": f1 for i, f1 in enumerate(splits_f1_score)},
            "average_f1": avg_f1
        }, f, indent=2)
    print(f"\n结果已保存至: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QFVS evaluation（按文件夹批量加载视频）')
    parser.add_argument("--work_dir", type=str, help="工作目录")
    parser.add_argument("--splits_file", type=str, help="分割文件路径")
    parser.add_argument("--mapping_file", type=str, help="映射文件路径")
    parser.add_argument("--Tags_file", type=str, help='Tags.mat文件路径')
    parser.add_argument("--gt_dir", type=str, help='Ground truth目录')
    parser.add_argument("--meta_data_dir", type=str, help="包含PredMetaData_1~4的根目录")
    parser.add_argument("--norm", type=str, choices=['None', 'MinMax', 'Exp', 'MinMax+Exp'], help="归一化方式")
    args = parser.parse_args()
    run(args)
