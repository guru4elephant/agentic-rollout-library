# import matplotlib.pyplot as plt
import numpy as np

import subprocess
import json
import tempfile
import os
import re


def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding = 'utf-8') as file:
         for line in file:
            line = line.strip()
            if line:
                data = json.loads(line)
                data_list.append(data)
    return data_list

def read_file(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as file:
        text = file.read()
    return text

def get_response(messages):
    config = {
        'location': 'your api',
        'header': 'Authorization: Bearer ... (your key)'
    }
    data_dict = {
        "model": "deepseek-v3-2-251201",
        "messages": messages,
        "temperature": 0.1,
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
    text_lines = text.split('\n')
    
    result = {}
    now_str = ""
    for line in text_lines:
        if line.startswith("** Thinking **") or line.startswith("**Thinking**"):
            now_str = line + '\n'
        # elif line.startswith("** LLM's erroneous patch **") or line.startswith("**LLM's erroneous patch**"):
        #     result['thinking'] = now_str
        #     now_str = line + '\n'
        # elif line.startswith("** Correct patch **") or line.startswith("**Correct patch**"):
        #     result['error_patch'] = now_str
        #     now_str = line + '\n'
        elif line.startswith("** Concise causes") or line.startswith("**Concise causes"):
            result['thinking'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Specific analysis") or line.startswith("**Specific analysis"):
            result['concise_cause'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Error type") or line.startswith("**Error type"):
            result['specific_analysis'] = now_str
            now_str = line + '\n'
        elif line.startswith("** Patch fix difficulty score **") or line.startswith("**Patch fix difficulty score**"):
            result['error_type'] = now_str
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
analysis_path = f'/root/sft_data/trajectory_analyzed_result/analyze/{model}/trajectory_analysis.jsonl'

def case_study():
    analysis_results = read_jsonl(analysis_path)
    for analysis_result in analysis_results:
        instance_id = analysis_result['instance_id']
        repo = analysis_result['repo']
        issue = analysis_result['issue']
        golden_patch = analysis_result['golden_patch']
        test_patch = analysis_result['test_patch']
        response = analysis_result['deepseek_response']
        
        if 'no error' in response.lower():
            print(f"[INSTANCE ID]: {instance_id}")
            print(f"[REPO]: {repo}")
            print(f"[ISSUE]:\n{issue}")
            print(f"[GOLDEN PATCH]:\n{golden_patch}")
            print(f"[TEST PATCH]:\n{test_patch}")
            print(f"[RESPONSE]:\n{response}")

            break


if __name__ == "__main__":
    case_study()