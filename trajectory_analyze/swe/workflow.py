import argparse
import os

import init_trajectory_analyzer
import deepseek_trajectory_analyzer
import deepseek_repo_analyzer

def main():
    """主函数"""
    # 输入参数：
    #    - 文件目录
    #    - query 500个问题集合路径，
    #    - result：结果文件路径
    #    - output_path：输出的文件夹路径

    parser = argparse.ArgumentParser(description='SWE轨迹分析')
    parser.add_argument('--directory', type=str, 
                        default='/root/sft_data/trajectory/swe/qianfan-coder/ds-swdata-filtered-iter-0000518-v1-output',
                        help='要分析的trajectory轨迹路径')
    parser.add_argument('--query_path', type=str,
                       default='/root/sft_data/trajectory/swe/query.jsonl',
                       help='SWE bench问题路径')    
    parser.add_argument('--result_path', type=str,
                       default='/root/sft_data/trajectory/swe/swe_result.jsonl',
                       help='SWE结果路径')
    parser.add_argument('--output_path', type=str,
                        default='/root/sft_data/trajectory_result/swe/qianfan-coder', # Change to your folder path
                        help='轨迹分析结果的存储路径')
    parser.add_argument('--deepseek_analyze', type=bool,
                        default=True,
                        help='是否选择使用Deepseek v3.2实现轨迹分析')
    parser.add_argument('--max_workers', type=int,
                        default=10,
                        help='调deepseek api的并发api数量')

    args = parser.parse_args()
    
    if not os.path.exists(args.output_path): os.makedirs(args.output_path)


    try:
        print(f"=========================开始阶段1轨迹分析=========================")
        init_trajectory_analyzer.main(
            # input_path = '/root/sft_data/trajectory/swe/qianfan-coder/ds-swdata-filtered-iter-0000518-v1-output',
            # output_path = '/root/sft_data/trajectory_result/swe/qianfan-coder',
            # query_path = '/root/sft_data/trajectory/swe/query.jsonl',
            input_path = args.directory,
            output_path = args.output_path,
            query_path = args.query_path
        )
        print(f"=========================阶段1轨迹分析完成=========================\n")

        if args.deepseek_analyze == False: 
            print('分析完成！')
            return 
        
        print(f"=========================开始阶段2轨迹分析=========================")
        # 对每条轨迹进行分析
        init_trajectory_analysis_path = f'{args.output_folder_path}/init_trajectory_analysis.jsonl'
        trajectory_path = f"{args.output_path}/init_trajectory_analysis.jsonl",
        deepseek_trajectory_analyzer.main(
            trajectory_path = init_trajectory_analysis_path,
            output_folder = args.output_path,
            max_workers = args.max_workers
        )
        deepseek_trajectory_path = f"{args.output_path}/deepseek_trajectory_analysis.jsonl"
        # 对分析后轨迹进行总结
        deepseek_repo_analyzer.main(
            result_path_str = args.result_path,
            input_path_str = deepseek_trajectory_path,
            output_path_str = args.output_path
        )
        print(f"=========================阶段2轨迹分析完成=========================\n")
        print('分析完成！')

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
 
 
if __name__ == "__main__":
    main()