from utils import read_jsonl


def cause_type_analyze(trajectorys: list,
                       output_folder: str):
    cause_type_list = ['correct implement',
                       'incorrect scoring of feature points', 
                       'ambiguous user requirement description', 
                       'incomplete implementation',
                       'incorrect implementation',
                       'incorrect tool use']
    results = {}
    for trajectory in trajectorys:
        if 'cause_type' not in trajectory: continue
        trajectory_cause_type = trajectory['cause_type']
        for cause_type in cause_type_list:
            if cause_type in trajectory_cause_type.lower():
                results[cause_type] = results.get(cause_type, 0) + 1

    print("===========Cause Type Analyze===========")
    print(f"轨迹数量：{len(trajectorys)}")
    for key in cause_type_list:
        print(f"{key}: {results.get(key,0)}条，占比{((results.get(key,0)/len(trajectorys))*100):.2f}%")
    print("")

    with open(f"{output_folder}/deepseek_trajectory_analysis.txt", "w", encoding = 'utf-8') as f:
        f.write("==============第二阶段分析结果============\n\n")
        f.write("===========Cause Type Analyze===========\n")
        for key in cause_type_list:
            f.write(f"{key}: {results.get(key,0)}条，占比{((results.get(key,0)/len(trajectorys))*100):.2f}%\n")
        f.write("\n")

    

def performance_type_analyze(trajectorys: list,
                             output_folder: str):
    performance_type_list = ['all feature point implement correctly',
                             'single feature: interface implementation error',
                             'single feature: interaction logic error',
                             'multiple features: interface implementation error',
                             'multiple features: interaction logic error',
                             'both interface implementation and interaction logic errors present']
    results = {}
    for trajectory in trajectorys:
        if 'performance_type' not in trajectory: continue
        trajectory_cause_type = trajectory['performance_type']
        for cause_type in performance_type_list:
            if cause_type in trajectory_cause_type.lower():
                results[cause_type] = results.get(cause_type, 0) + 1

    print("===========Performance Type Analyze===========")
    print(f"轨迹数量：{len(trajectorys)}")
    for key in performance_type_list:
        print(f"{key}: {results.get(key,0)}条，占比{((results.get(key,0)/len(trajectorys))*100):.2f}%")
    print("")

    with open(f"{output_folder}/deepseek_trajectory_analysis.txt", "a", encoding = 'utf-8') as f:
        f.write("===========Performance Type Analyze===========\n")
        for key in performance_type_list:
            f.write(f"{key}: {results.get(key,0)}条，占比{((results.get(key,0)/len(trajectorys))*100):.2f}%\n")
        f.write("\n")


def fix_difficulty_score_analyze(trajectorys: list,
                                 output_folder: str):
    performance_type_list = ['1', '2', '3', '4', '5']
    results = {}
    for trajectory in trajectorys:
        if 'difficulty_score' not in trajectory: continue
        trajectory_cause_type = trajectory['difficulty_score']
        for cause_type in performance_type_list:
            if cause_type in trajectory_cause_type.lower():
                results[cause_type] = results.get(cause_type, 0) + 1

    print("===========Fix Difficulity Score Analyze===========")
    print(f"轨迹数量：{len(trajectorys)}")
    for key in performance_type_list:
        print(f"{key}分: {results.get(key,0)}条，占比{((results.get(key,0)/len(trajectorys))*100):.2f}%")
    print("")

    with open(f"{output_folder}/deepseek_trajectory_analysis.txt", "a", encoding = 'utf-8') as f:
        f.write("===========Fix Difficulity Score Analyze===========\n")
        for key in performance_type_list:
            f.write(f"{key}分: {results.get(key,0)}条，占比{((results.get(key,0)/len(trajectorys))*100):.2f}%\n")
        f.write("\n")

def main(trajectory_path: str, output_folder: str):
    trajectorys = read_jsonl(trajectory_path)

    cause_type_analyze(trajectorys, output_folder)
    performance_type_analyze(trajectorys, output_folder)
    fix_difficulty_score_analyze(trajectorys, output_folder)

if __name__ == "__main__":
    main(
        trajectory_path = '/root/sft_data/trajectory_result/miaoda/analyze/qianfan-coder/deepseek_trajectory_analysis.jsonl',
        output_folder = '/root/sft_data/trajectory_result/miaoda/analyze/qianfan-coder'
    )
