import os
import json

def get_json_files(folder):
    return {f for f in os.listdir(folder) if f.startswith("episode_") and f.endswith("_final_res.json")}

def count_positive_rewards(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Assumes 'reward' is either at the top level or inside a list
    if isinstance(data, dict):
        return int(data.get("reward", 0) > 0)
    elif isinstance(data, list):
        return sum(1 for item in data if isinstance(item, dict) and item.get("reward", 0) > 0)
    return 0

def analyze_folders(folder1, folder2):
    files1 = get_json_files(folder1)
    files2 = get_json_files(folder2)

    overlapping_files = files1 & files2
    print(f"Found {len(overlapping_files)} overlapping files.")

    total_correct1 = []
    total_correct2 = []
    
    for file_name in sorted(overlapping_files):
        path1 = os.path.join(folder1, file_name)
        path2 = os.path.join(folder2, file_name)

        count1 = count_positive_rewards(path1)
        count2 = count_positive_rewards(path2)

        if count1 > 0:
            total_correct1.append(file_name)
        if count2 > 0:
            total_correct2.append(file_name)
        
    total_correct1 = set(total_correct1)
    total_correct2 = set(total_correct2)
    suc1 = total_correct1 - total_correct2
    suc2 = total_correct2 - total_correct1
    
    print(f"  {folder1}: {len(total_correct1)} success, total {len(overlapping_files)}")
    print(f"unique success in {suc1}")
    print(f"  {folder2}: {len(total_correct2)} success, total {len(overlapping_files)}")
    print(f"unique success in {suc2}")
    print("end")


#'eval_sets': ['base', 'common_sense', 'complex_instruction', 'spatial', 'visual_appearance', 'long_horizon'],

# Example usage:
#   'eval_sets': ['base', 'common_sense', 'long_horizon', 'visual_appearance', 'complex_instruction'], 
# path1 = "/home/jianih/research/EmbodiedBench/running/eb_nav/gpt-4o_baseline/visual_appearance/results"
path1 = "/home/jianih/research/EmbodiedBench/running/eb_nav/gpt-4o_gd_2/visual_appearance/results"
# path2 = "/home/jianih/research/EmbodiedBench/running/eb_nav/Qwen2.5-VL-72B-Instruct_gd_laser_topk_3/visual_appearance/results"

# path2 = "/home/jianih/research/EmbodiedBench/running/eb_nav/gpt-4o_gd_laser_topk_3/base/results"
# path2 = "/home/jianih/research/EmbodiedBench/running/eb_nav/gpt-4o_gd_laser/common_sense/results"
# path2 = "/home/jianih/research/EmbodiedBench/running/eb_nav/gpt-4o_gd_laser_topk_1/long_horizon/results"
path2 = "/home/jianih/research/EmbodiedBench/running/eb_nav/gpt-4o_gd_laser_topk_3/visual_appearance/results"
# path2 = "/home/jianih/research/EmbodiedBench/running/eb_nav/gpt-4o_gd_laser/complex_instruction/results"

analyze_folders(path1, path2)
print("end")
