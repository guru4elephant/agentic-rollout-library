import pandas as pd
from utils import read_jsonl
import json

pd.set_option('display.float_format', lambda x: '%.3f' % x)
repos = ['all', 'astropy', 'django', 'matplotlib', 'requests', 'xarray', 'pylint', 'pytest', 'scikit-learn', 'sphinx', 'sympy']

error_type_list = [
              'correct implement',
              'incorrect understanding of the issue requirement',
              'misunderstanding of code structure',
              'incomplete consideration of code structure',
              'overengineering solution',
              'incorrect tool use',
              'incomplete consideration of boundary cases'
              ]

human_difficulity_list = ['<15 min fix', '15 min - 1 hour', '1-4 hours', '>4 hours']
difficulity_score_list = [1, 2, 3, 4, 5]

# result_path = f'/root/sft_data/trajectory/swe/swe_result.jsonl'
# input_path = f"/root/sft_data/trajectory_result/swe/qianfan-coder/deepseek_trajectory_analysis.jsonl"
# output_path = f"/root/sft_data/trajectory_result/swe/qianfan-coder"
result_path, input_path, output_path = "", "", ""
unresolved_ids = []

def error_type_analysis(instance_results: list, repo: str):
    filtered_results = []
    for instance_result in instance_results:
        if repo == 'all' or instance_result['repo'] == repo:
            filtered_results.append(instance_result)
    
    result = []
    result.append(repo)
    result.append(len(filtered_results))

    for error_type in error_type_list:
        count = 0
        if error_type == 'correct implement':
            for instance_result in filtered_results:
                if instance_result['instance_id'] not in unresolved_ids:
                    count += 1
            result.append(str(f"{count}({(count * 100.0 / len(filtered_results)):.1f}%)"))
        else:
            for instance_result in filtered_results:
                if instance_result['instance_id'] not in unresolved_ids: continue
                if error_type in instance_result['error_type'].lower():
                    count += 1    
            result.append(str(f"{count}({(count * 100.0 / len(filtered_results)):.1f}%)"))
    return result
    
def repo_error_type_analysis():
    instance_results = read_jsonl(input_path)

    repo_error_type_results = []
    for repo in repos:
        repo_error_type_result = error_type_analysis(instance_results, repo)
        repo_error_type_results.append(repo_error_type_result)
    
    columns = ['repo', 'trajectory count']
    columns.extend(error_type_list)

    df = pd.DataFrame(repo_error_type_results, columns=columns)
    print("=====================SWE Repo Error Type分析=====================")
    print(df)

    with open(f"{output_path}/deepseek_trajectory_analysis.txt", 'w') as file: 
        file.write("=====================SWE Repo Error Type分析=====================\n")
    df.to_csv(f"{output_path}/deepseek_trajectory_analysis.txt", mode = 'a', sep = '\t', index = False, encoding = 'utf-8')

def human_difficulity_analysis(instance_results: list, repo: str):
    filtered_results = []
    for instance_result in instance_results:
        if instance_result['instance_id'] not in unresolved_ids: continue
        if repo == 'all' or instance_result['repo'] == repo:
            filtered_results.append(instance_result)
    
    result = []
    result.append(repo)
    result.append(len(filtered_results))

    for human_difficulity in human_difficulity_list:
        count = 0
        for instance_result in filtered_results:
            if instance_result['instance_id'] not in unresolved_ids: continue
            if human_difficulity in instance_result['human_difficulty']:
                count += 1    
        result.append(str(f"{count}({(count * 100.0 / len(filtered_results)):.1f}%)"))
    
    return result

def repo_human_difficulity_analysis():
    instance_results = read_jsonl(input_path)

    repo_human_difficulity_results = []
    for repo in repos:
        repo_human_difficulity_result = human_difficulity_analysis(instance_results, repo)
        repo_human_difficulity_results.append(repo_human_difficulity_result)
    
    columns = ['repo', '错误轨迹数量']
    columns.extend(human_difficulity_list)

    df = pd.DataFrame(repo_human_difficulity_results, columns=columns)
    print("=====================SWE Repo Human Difficulity分析=====================")
    print(df)

    with open(f"{output_path}/deepseek_trajectory_analysis.txt", 'a') as file: 
        file.write("\n=====================SWE Repo Human Difficulity分析=====================\n")
    df.to_csv(f"{output_path}/deepseek_trajectory_analysis.txt", mode = 'a', sep = '\t', index = False, encoding = 'utf-8')

def difficulity_score_analysis(instance_results: list, repo: str):
    filtered_results = []
    for instance_result in instance_results:
        if instance_result['instance_id'] not in unresolved_ids: continue
        if repo == 'all' or instance_result['repo'] == repo:
            filtered_results.append(instance_result)
    
    result = []
    result.append(repo)
    result.append(len(filtered_results))
    
    avg_score = 0
    for difficulity_score in difficulity_score_list:
        count = 0
        for instance_result in filtered_results:
            if str(difficulity_score) in instance_result['difficulty_score']:
                count += 1 
                avg_score += int(difficulity_score)
     
        result.append(str(f"{count}({(count * 100.0 / len(filtered_results)):.1f}%)"))

    result.append(avg_score / len(filtered_results))
    return result

def repo_difficulity_score_analysis():
    instance_results = read_jsonl(input_path)

    repo_difficulity_score_results = []
    for repo in repos:
        repo_difficulity_score_result = difficulity_score_analysis(instance_results, repo)
        repo_difficulity_score_results.append(repo_difficulity_score_result)
    
    columns = ['repo', '错误轨迹数量']
    columns.extend(difficulity_score_list)
    columns.append('avg.')

    df = pd.DataFrame(repo_difficulity_score_results, columns=columns)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    print("=====================SWE Repo Difficulity Score分析=====================")
    print(df)

    with open(f"{output_path}/deepseek_trajectory_analysis.txt", 'a') as file: 
        file.write("\n=====================SWE Repo Difficulity Score分析=====================\n")
    df.to_csv(f"{output_path}/deepseek_trajectory_analysis.txt", mode = 'a', sep = '\t', index = False, encoding = 'utf-8')

def main(input_path_str: str, result_path_str: str, output_path_str: str):
    
    global input_path, result_path, output_path
    input_path, result_path, output_path = input_path_str, result_path_str, output_path_str
    with open(result_path, 'r', encoding = 'utf-8') as file: result = json.load(file)
    global unresolved_ids
    unresolved_ids = result['unresolved_ids']
    repo_error_type_analysis()
    repo_human_difficulity_analysis()
    repo_difficulity_score_analysis()
    # 过滤
   

if __name__ == "__main__":
    main(
        result_path_str = f'/root/sft_data/trajectory/swe/swe_result.jsonl',
        input_path_str = f"/root/sft_data/trajectory_result/swe/qianfan-coder/deepseek_trajectory_analysis.jsonl",
        output_path_str = f"/root/sft_data/trajectory_result/swe/qianfan-coder"
    )
    
    

