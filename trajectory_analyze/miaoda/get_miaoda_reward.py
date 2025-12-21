import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import init_trajectory_analyzer
import deepseek_trajectory_analyzer
import deepseek_trajectory_statistics

def load_dataset_filter_info(dataset_path: str) -> Dict[str, List[int]]:
    """
    从数据集文件中加载filter_list信息
 
    参数:
        dataset_path: 数据集文件路径（JSONL格式）
 
    返回:
        字典，key为qid，value为filter_list
    """
    filter_dict = {}
 
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    qid = data.get('qid')
                    filter_list = data.get('filter_list', [])
                    if qid:
                        filter_dict[qid] = filter_list
    except Exception as e:
        print(f"读取数据集文件失败: {e}")
        raise
 
    return filter_dict
 
 
def calculate_filtered_reward(scores: List[int], filter_list: List[int]) -> float:
    """
    根据filter_list计算过滤后的分数平均值
 
    参数:
        scores: 分数列表
        filter_list: 过滤列表，1表示参与计算，0表示不参与
 
    返回:
        过滤后的平均分数
    """
    if len(scores) != len(filter_list):
        print(f"警告: scores长度({len(scores)})与filter_list长度({len(filter_list)})不匹配")
        # 取最小长度
        min_len = min(len(scores), len(filter_list))
        scores = scores[:min_len]
        filter_list = filter_list[:min_len]
 
    # 根据filter_list过滤scores
    filtered_scores = [score for score, filter_val in zip(scores, filter_list) if filter_val == 1]
 
    if not filtered_scores:
        return 0.0
 
    return sum(filtered_scores) / len(filtered_scores)
 
 
def is_repo_perfect(scores: List[int], filter_list: List[int]) -> bool:
    """
    检查repo是否完全正确（所有有效功能点都是1分）
 
    参数:
        scores: 分数列表
        filter_list: 过滤列表，1表示参与计算，0表示不参与
 
    返回:
        True如果所有有效功能点都是1分，False否则
    """
    if len(scores) != len(filter_list):
        # 取最小长度
        min_len = min(len(scores), len(filter_list))
        scores = scores[:min_len]
        filter_list = filter_list[:min_len]
 
    # 获取所有有效功能点（filter_list中为1的）
    valid_scores = [score for score, filter_val in zip(scores, filter_list) if filter_val == 1]
 
    # 如果没有有效功能点，返回False
    if not valid_scores:
        return False
 
    # 检查是否所有有效功能点都是1分
    return all(score == 1 for score in valid_scores)
 
 
def calculate_reward_average(directory_path: str, dataset_path: str) -> Tuple[float, Dict]:
    """
    计算指定目录下所有以'reward'结尾的JSON文件中reward分数的平均值
    基于数据集中的filter_list进行过滤计算
 
    参数:
        directory_path: 目录路径
        dataset_path: 数据集文件路径
 
    返回:
        (总平均分, 统计信息字典)
    """
    dir_path = Path(directory_path)
 
    # 检查目录是否存在
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory_path}")
 
    if not dir_path.is_dir():
        raise NotADirectoryError(f"路径不是目录: {directory_path}")
 
    # 加载数据集的filter信息
    print(f"正在加载数据集文件: {dataset_path}")
    filter_dict = load_dataset_filter_info(dataset_path)
    print(f"已加载 {len(filter_dict)} 个样本的filter信息\n")
 
    repo_scores: List[float] = []
    perfect_repos = 0  # 完全正确的repo数量
    total_valid_repos = 0  # 有效的repo总数
    processed_files = 0
    error_files = []
    skipped_files = []
 
    # 遍历目录下所有以'reward'结尾的文件
    for file_path in dir_path.iterdir():
        if file_path.is_file() and str(file_path).endswith('.reward'):
            processed_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
 
                    # 获取qid和scores
                    qid = data.get('instance_id') or file_path.stem
                    scores = data.get('scores', [])
 
                    # 获取对应的filter_list
                    if qid not in filter_dict:
                        skipped_files.append((file_path.name, f"在数据集中未找到qid={qid}"))
                        continue
 
                    filter_list = filter_dict[qid]
 
                    # 如果filter_list全为0，跳过此文件
                    if sum(filter_list) == 0:
                        skipped_files.append((file_path.name, "filter_list全为0，跳过"))
                        continue
 
                    # 计算过滤后的平均分
                    filtered_avg = calculate_filtered_reward(scores, filter_list)
                    repo_scores.append(filtered_avg)
 
                    # 检查是否完全正确（所有有效功能点都是1分）
                    is_perfect = is_repo_perfect(scores, filter_list)
                    total_valid_repos += 1
                    if is_perfect:
                        perfect_repos += 1
 
                    print(f"文件: {file_path.name}, qid: {qid}, "
                          f"scores: {scores}, filter_list: {filter_list}, "
                          f"过滤后平均分: {filtered_avg:.4f}, 完全正确: {'是' if is_perfect else '否'}")
 
            except json.JSONDecodeError:
                error_files.append((file_path.name, "JSON解析失败"))
            except (ValueError, TypeError) as e:
                error_files.append((file_path.name, f"数值转换失败: {e}"))
            except Exception as e:
                error_files.append((file_path.name, f"读取错误: {e}"))
 
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"处理的文件总数: {processed_files}")
    print(f"成功计算的文件数: {len(repo_scores)}")
    print(f"跳过的文件数: {len(skipped_files)}")
    print(f"处理失败的文件数: {len(error_files)}")
 
    if skipped_files:
        print(f"\n跳过的文件 ({len(skipped_files)}):")
        for filename, reason in skipped_files:
            print(f"  - {filename}: {reason}")
 
    if error_files:
        print(f"\n处理失败的文件 ({len(error_files)}):")
        for filename, error in error_files:
            print(f"  - {filename}: {error}")
 
    # 计算最终平均值和完全正确准确率
    perfect_accuracy = perfect_repos / total_valid_repos if total_valid_repos > 0 else 0.0
 
    stats = {
        'total_files': processed_files,
        'successful_files': len(repo_scores),
        'skipped_files': len(skipped_files),
        'error_files': len(error_files),
        'individual_scores': repo_scores,
        'perfect_repos': perfect_repos,
        'total_valid_repos': total_valid_repos,
        'perfect_accuracy': perfect_accuracy
    }
 
    if repo_scores:
        average = sum(repo_scores) / len(repo_scores)
        print(f"\n{'='*60}")
        print(f"指标1 - 平均准确率（按功能点计算）:")
        print(f"  {average:.4f} ({average*100:.2f}%)")
        print(f"\n指标2 - 完全正确准确率（所有功能点都是1分）:")
        print(f"  {perfect_accuracy:.4f} ({perfect_accuracy*100:.2f}%)")
        print(f"  完全正确的repo: {perfect_repos}/{total_valid_repos}")
        print(f"{'='*60}")
        return average, stats
    else:
        print("\n未找到有效的reward数据")
        return 0.0, stats
 
 
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='计算reward分数的平均值（基于filter_list过滤）')
    parser.add_argument('--directory', type=str, 
                        default='/root/sft_data/trajectory/miaoda/qianfan-coder/ds-swdata-filtered-iter-0000518-v3-output',
                        help='要计算分数的目录路径')
    parser.add_argument('--dataset', type=str,
                       default='/mnt/cfs_bj_mt/workspace/tianlun-2/grpo_train/claude/agent_rollout_eval/agentic-rollout-library-refactor/miaoda-dataset/244-sample-converted-format-v2.jsonl',
                       help='数据集文件路径（默认: miaoda-dataset/244-sample-converted-format-v2.jsonl）')    
    # add 
    parser.add_argument('--output_folder_path', type=str,
                        default='/root/sft_data/trajectory_result/miaoda/analyze/qianfan-coder', # Change to your folder path
                        help='轨迹分析结果的存储目录')
    parser.add_argument('--deepseek_analyze', type=bool,
                        default=True,
                        help='是否选择使用Deepseek v3.2实现轨迹分析')
    parser.add_argument('--max_workers', type=int,
                        default=10,
                        help='调deepseek api的并发api数量')

    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder_path): os.makedirs(args.output_folder_path)


    try:
        avg_reward, stats = calculate_reward_average(args.directory, args.dataset)
        perfect_acc = stats.get('perfect_accuracy', 0.0)
        print(f"\n处理完成！")
        print(f"指标1 - 平均准确率: {avg_reward:.4f} ({avg_reward*100:.2f}%)")
        print(f"指标2 - 完全正确准确率: {perfect_acc:.4f} ({perfect_acc*100:.2f}%)")

        print(f"=========================开始阶段1轨迹分析=========================")
        init_trajectory_analyzer.main(
            miaoda_input_path = args.directory,
            miaoda_output_path = args.output_folder_path,
            query_path = args.dataset
        )
        print(f"=========================阶段1轨迹分析完成=========================\n")

        if args.deepseek_analyze == False: 
            print('分析完成！')
            return 
        
        print(f"=========================开始阶段2轨迹分析=========================")
        # 对每条轨迹进行分析
        init_trajectory_analysis_path = f'{args.output_folder_path}/init_trajectory_analysis.jsonl'
        deepseek_trajectory_analyzer.main(
            trajectory_path = init_trajectory_analysis_path,
            output_folder = args.output_folder_path,
            max_workers = args.max_workers
        )
        # 对分析后轨迹进行总结
        deepseek_trajectory_analysis_path = f'{args.output_folder_path}/deepseek_trajectory_analysis.jsonl'
        deepseek_trajectory_statistics.main(
            trajectory_path = deepseek_trajectory_analysis_path,
            output_folder = args.output_folder_path
        )
        print(f"=========================阶段2轨迹分析完成=========================\n")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
 
 
if __name__ == "__main__":
    main()