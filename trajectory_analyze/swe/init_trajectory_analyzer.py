#!/usr/bin/env python3
import json
import os
import sys
from typing import Dict, Any, List
import numpy as np
import re
import pandas as pd
from utils import read_jsonl, read_file

# input_path = '/root/sft_data/trajectory/swe/qianfan-coder/ds-swdata-filtered-iter-0000518-v1-output'
# output_path = '/root/sft_data/trajectory_result/swe/qianfan-coder'
# query_path = '/root/sft_data/trajectory/swe/query.jsonl'


repos = ['all', 'astropy', 'django', 'matplotlib', 'requests', 'xarray', 'pylint', 'pytest', 'scikit-learn', 'sphinx', 'sympy']
columns = [
           'Repo', '轨迹数量', '平均迭代轮数', '平均响应长度', '超长个数', '超长占比', 
           '平均工具调用错误次数', '平均工具调用错误占比',
           '平均file editor出现错误次数', '平均file editor出现错误占比',
           '平均bash执行错误次数', '平均bash执行错误占比'
          ]
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def analyze_trajectory(trajectory: list,
                       patch: str,
                       query: dict
                       ) -> Dict[str, Any]:
    """分析单个jsonl文件的迭代轮数和长度"""
    try:
        # trajectory = []
        # with open(file_path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         if line.strip():
        #             trajectory.append(json.loads(line.strip()))
               
        instance_id = query['instance_id']
        repo_id = query['repo'].split('/')[1]
        golden_patch = query['patch']
        test_patch = query['test_patch']
        issue = query['problem_statement']
        hints_text = query['hints_text']
  
        # if 'astropy__astropy-7336' != instance_id: return {}
        # print(instance_id)

        tool_sequence = ""
        for msg in trajectory:
            content = msg.get('content')
            if msg.get('role') == 'assistant':
                pattern = r'<function=.*?</function>'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    tool_sequence += "[Tool Call]:\n" + match.group(0) + '\n'

            if msg.get('role') == 'user' and 'Execution output of' in content:
                tool_sequence += "[Tool Result]:\n" + content + '\n'

        assistant_messages = [msg.get('content') for msg in trajectory if msg.get('role') == 'assistant']
        user_messages = [msg.get('content') for msg in trajectory if msg.get('role') == 'user'] 

        if user_messages == []: return {}
        del user_messages[0]
        
        assistant_rounds = sum(1 for msg in trajectory if msg.get('role') == 'assistant')
        assistant_msg_len = [len(msg) for msg in assistant_messages]
        response_avg_length = sum(assistant_msg_len) / len(assistant_msg_len) if assistant_msg_len else 0

        error_count = 0
        error_dict = {}
        for idx in range(len(user_messages)):
            assistant_message, user_message = assistant_messages[idx], user_messages[idx]
            pattern = r'<function=([^>]+)>'
            match = re.findall(pattern, assistant_message)
            tool_name = match[0].strip()
            if '[STDOUT]:\nERROR:' in user_message or 'Error executing command:' in user_message:
                error_count += 1
                error_dict[tool_name] = error_dict.get(tool_name, 0) + 1
        
        over_length = False
        if assistant_rounds >= 100: over_length = True
        
        result = {
            'instance_id': instance_id,
            'repo_id': repo_id,
            'issue': issue,
            'hints_text': hints_text,
            'llm_patch': patch,
            'golden_patch': golden_patch,
            'test_patch': test_patch,
            'human_difficulty': query['difficulty'],

            'rounds': assistant_rounds,                  #迭代轮数
            'response_avg_length': response_avg_length,  #平均响应长度
            'extra_long': over_length,                   #是否迭代未停止
            'tool_sequence': tool_sequence,              #工具调用序列

            'tool_error_count': error_count,
            'tool_error_ratio': error_count / assistant_rounds,
        }
        for key, value in error_dict.items():
            result[f'{key}_error_count'] = value
            result[f'{key}_error_ratio'] = value / assistant_rounds
        # print(result['tool_error_count'])
        return result
    except Exception as e:
        print(f"❌ 处理 {instance_id} 时出错: {e}")
        return None

def analyze_repo(results: list, repo: str):
    repo_result = []

    filter_results = []
    for result in results:
        if repo == 'all': filter_results.append(result)
        if result['repo_id'] == repo: filter_results.append(result)
    
    #基于columns计算各个指标
    repo_result.append(repo)
    repo_result.append(len(filter_results))

    rounds = [result['rounds'] for result in filter_results]
    response_length = [result['response_avg_length'] for result in filter_results]
    extra_long = [result['extra_long'] for result in filter_results]

    tool_error_count = [result.get('tool_error_count', 0) for result in filter_results]
    tool_error_ratio = [result.get('tool_error_ratio', 0) for result in filter_results]
    
    file_error_count = [result.get('file_editor_error_count', 0) for result in filter_results]
    file_error_ratio = [result.get('file_editor_error_ratio', 0) for result in filter_results]
    
    bash_error_count = [result.get('execute_bash_error_count', 0) for result in filter_results]
    bash_error_ratio = [result.get('execute_bash_error_ratio', 0) for result in filter_results]


    repo_result.append(np.mean(rounds))
    repo_result.append(np.mean(response_length))
    repo_result.append(sum(extra_long))
    repo_result.append(sum(extra_long) / len(filter_results))

    repo_result.append(np.mean(tool_error_count))
    repo_result.append(np.mean(tool_error_ratio))
    

    repo_result.append(np.mean(file_error_count))
    repo_result.append(np.mean(file_error_ratio))

    repo_result.append(np.mean(bash_error_count))
    repo_result.append(np.mean(bash_error_ratio))

    return repo_result


def write_result(trajectory_results, repo_results, output_path):
    if not os.path.exists(output_path): os.makedirs(output_path)

    with open(f"{output_path}/init_trajectory_analysis.jsonl", 'w', encoding = 'utf-8') as f: 
        for line in trajectory_results:
            if line: 
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    print("================SWE初步分析结果===============")
    df = pd.DataFrame(repo_results, columns=columns)
    print(df)

    # repo_results_df.to_excel(f"{output_path}/init_repo_analysis.xlsx", index=False)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    with open(f"{output_path}/init_trajectory_analysis.txt", 'w') as file: 
        file.write(("================SWE初步分析结果===============\n"))
    df.to_csv(f"{output_path}/init_trajectory_analysis.txt", mode = 'a', sep = '\t', index = False, encoding = 'utf-8')

def main(query_path: str, input_path: str, output_path: str):
    querys = read_jsonl(query_path)
    trajectory_results, repo_results = [], []
    for query in querys:
        instance_id = query['instance_id']
        if not os.path.exists(f'{input_path}/{instance_id}.context'): continue
        if not os.path.exists(f'{input_path}/{instance_id}.patch'): continue

        trajectory = read_jsonl(f'{input_path}/{instance_id}.context')
        patch = read_file(f'{input_path}/{instance_id}.patch')
        trajectory_results.append(analyze_trajectory(trajectory, patch, query))

    for repo in repos:
        repo_results.append(analyze_repo(trajectory_results, repo))
    
    write_result(trajectory_results, repo_results, output_path)
        
if __name__ == "__main__":
    main(
        input_path = '/root/sft_data/trajectory/swe/qianfan-coder/ds-swdata-filtered-iter-0000518-v1-output',
        output_path = '/root/sft_data/trajectory_result/swe/qianfan-coder',
        query_path = '/root/sft_data/trajectory/swe/query.jsonl',
    )