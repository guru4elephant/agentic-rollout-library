# import matplotlib.pyplot as plt
import numpy as np

import subprocess
import json
import tempfile
import os
import re


def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
         for line in file:
            line = line.strip()
            if line:
                data = json.loads(line)
                data_list.append(data)
    return data_list

def get_response(messages):
    config = {
        'location': 'http://211.23.3.237:27544/v1/chat/completions',
        'header': 'Authorization: Bearer sk-qq7xJtnAdB1Gv6IkHTQhDAPuUAT700vF3CMmGinILsmP2HuY'
    }
    data_dict = {
        "model": "deepseek-v3-2-251201",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 32768,
        "thinking": {"type": "disabled"}
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        json.dump(data_dict, temp_file, ensure_ascii=False, indent=2)
        temp_file_path = temp_file.name
    
    try:
        curl_command = [
            'curl', '--location', config['location'],
            '--header', 'Content-Type: application/json',
            '--header', config['header'],
            '--data', f'@{temp_file_path}'
        ]
        
        result = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            check=True
        )
        response = json.loads(result.stdout)
        
        return response['choices'][0]['message']['content']
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from API: {e}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to call API: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def parse_response(text: str):
    text = text.strip()
    result = {}
    result['response'] = text

    text_lines = text.split('\n')
    
    now_str = ""
    for line in text_lines:
        if line.startswith("** Thinking **") or line.startswith("**Thinking**"):
            now_str = line + '\n'
        elif line.startswith("** Shortcoming of LLM Patch **") or line.startswith("**Shortcoming of LLM Patch**"):
            result['thinking'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Root causes of LLM patch shortcoming **") or line.startswith("**Root causes of LLM patch shortcoming**"):
            result['shortcoming'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Incorrect cause type of LLM Patch") or line.startswith("**Incorrect cause type of LLM Patch"):
            result['root_cause'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Explanation for choosing this cause type") or line.startswith("**Explanation for choosing this cause type"):
            result['cause_type'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Incorrect performance type of LLM Patche") or line.startswith("**Incorrect performance type of LLM Patch"):
            result['cause_explanation'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Explanation for choosing this performace type **") or line.startswith("**Explanation for choosing this performace type**"):
            result['performance_type'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Patch fix difficulty score **") or line.startswith("**Patch fix difficulty score**"):
            result['performance_explanation'] = now_str
            now_str = line + '\n'
        else:
            now_str += line + '\n'
    
    result['difficulty_score'] = now_str
    return result

# def draw_4(data: list, name: str, model: str):
#     min_val = np.min(data)
#     max_val = np.max(data)
#     interval = (max_val - min_val) / 4
#     bins = [min_val, min_val+interval, min_val+2*interval, min_val+3*interval, max_val]
#     counts, _ = np.histogram(data, bins=bins)
#     print(counts)
#     print(len(data))
#     proportions = counts / len(data) * 100
#     labels = [f"[{bins[i]:.1f}, {bins[i+1]:.1f}]" for i in range(4)]

#     # 绘制柱状图（占比）
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(labels, proportions, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
#     for bar, prop in zip(bars, proportions):
#         plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
#                 f'{prop:.1f}%', ha='center', va='bottom')

#     plt.title(f'{name.replace(".png","")} Proportion by Four Equal Intervals ({model})', fontsize=14)
#     plt.xlabel('Interval', fontsize=12)
#     plt.ylabel('Proportion (%)', fontsize=12)
#     plt.xticks(rotation=15)
#     plt.ylim(0, max(proportions) * 1.2)  # 调整Y轴范围
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig(name, dpi=300, bbox_inches='tight')

model = 'glm4.6'
analysis_path = f'/root/sft_data/trajectory_result/miaoda/analyze/qianfan-coder/deepseek_trajectory_analysis.jsonl'

def case_study():
    analysis_results = read_jsonl(analysis_path)
    for analysis_result in analysis_results[50:]:
        instance_id = analysis_result['instance_id']
        user_requirement = analysis_result['user_requirement']
        response = analysis_result['response']
        patch = analysis_result['patch']
        cause_type = analysis_result['cause_type']
        if 'incomplete consideration of code structure' in cause_type.lower():
            print(instance_id)
            print(user_requirement)
            print(response)
            # print(patch)
        
        


if __name__ == "__main__":
    # messages = [
    #     {'role': 'user', 'content': 'hello!'}
    # ]
    # print(get_response(messages))
    case_study()
    