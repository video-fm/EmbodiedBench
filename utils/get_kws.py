import os
import json
import glob
from collections import defaultdict
from typing import Dict, List, Any


def calculate_keyword_stats(exp_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Calculate keyword statistics for each task in an experiment.
    
    Args:
        exp_name: The experiment name (e.g., 'gpt-4o_gd_laser_topk_3')
    
    Returns:
        Dictionary with structure:
        {
            <TASK>: {
                "avg_keywords_per_request": {
                    "target_name": float,
                    "target_attribute": float,
                    "target_relation": float,
                    "related_objects": float
                },
                "unique_keywords_total": {
                    "target_name": int,
                    "target_attribute": int,
                    "target_relation": int,
                    "related_objects": int
                },
                "total_requests": int
            }
        }
    """
    base_path = f"/home/jianih/research/EmbodiedBench/running/eb_nav/{exp_name}"
    
    if not os.path.exists(base_path):
        raise ValueError(f"Experiment path does not exist: {base_path}")
    
    result = {}
    
    # Get all task directories
    task_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d))]
    
    for task in task_dirs:
        task_path = os.path.join(base_path, task)
        
        # Track keywords per request and unique keywords
        keyword_counts_per_request = []  # List of dicts, one per request
        unique_keywords = {
            "target_name": set(),
            "target_attribute": set(), 
            "target_relation": set(),
            "related_objects": set()
        }
        
        total_requests = 0
        
        # Find all episode JSON files
        episode_files = glob.glob(os.path.join(task_path, "episode_*.json"))
        
        for episode_file in episode_files:
            try:
                with open(episode_file, 'r') as f:
                    # Read each line as a separate JSON object
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            step_data = json.loads(line)
                            
                            # Extract scene_graph if it exists
                            if ('reasoning' in step_data and 
                                'scene_graph' in step_data['reasoning']):
                                
                                scene_graph = step_data['reasoning']['scene_graph']
                                total_requests += 1
                                
                                # Count keywords for this request
                                request_counts = {
                                    "target_name": 0,
                                    "target_attribute": 0,
                                    "target_relation": 0,
                                    "related_objects": 0
                                }
                                
                                for key in request_counts.keys():
                                    if key in scene_graph:
                                        values = scene_graph[key]
                                        if isinstance(values, list):
                                            request_counts[key] = len(values)
                                            # Add to unique set
                                            for item in values:
                                                if isinstance(item, list):
                                                    unique_keywords[key].add(str(item))
                                                else:
                                                    unique_keywords[key].add(item)
                                        else:
                                            request_counts[key] = 1
                                            unique_keywords[key].add(values)
                                
                                keyword_counts_per_request.append(request_counts)
                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
                            
            except Exception as e:
                print(f"Warning: Could not process {episode_file}: {e}")
                continue
        
        # Calculate averages
        avg_keywords = {
            "target_name": 0.0,
            "target_attribute": 0.0,
            "target_relation": 0.0,
            "related_objects": 0.0
        }
        
        if total_requests > 0:
            for key in avg_keywords.keys():
                total_for_key = sum(req[key] for req in keyword_counts_per_request)
                avg_keywords[key] = total_for_key / total_requests
        
        # Get unique counts
        unique_counts = {key: len(values) for key, values in unique_keywords.items()}
        
        result[task] = {
            "avg_keywords_per_request": avg_keywords,
            "unique_keywords_total": unique_counts,
            "total_requests": total_requests
        }
    
    return result


def calculate_global_keyword_stats(exp_name: str) -> Dict[str, Any]:
    """
    Calculate global keyword statistics across all tasks in an experiment.
    
    Args:
        exp_name: The experiment name (e.g., 'gpt-4o_gd_laser_topk_3')
    
    Returns:
        Dictionary with structure:
        {
            "global_unique_keywords": {
                "target_name": int,
                "target_attribute": int,
                "target_relation": int,
                "related_objects": int
            },
            "global_avg_keywords_per_request": {
                "target_name": float,
                "target_attribute": float,
                "target_relation": float,
                "related_objects": float
            },
            "total_requests_all_tasks": int,
            "per_task_stats": dict  # Same as calculate_keyword_stats output
        }
    """
    # Get per-task stats first
    per_task_stats = calculate_keyword_stats(exp_name)
    
    # Aggregate global unique keywords across all tasks
    global_unique_keywords = {
        "target_name": set(),
        "target_attribute": set(),
        "target_relation": set(),
        "related_objects": set()
    }
    
    # Track total counts for global averages
    global_keyword_counts = {
        "target_name": 0,
        "target_attribute": 0,
        "target_relation": 0,
        "related_objects": 0
    }
    
    total_requests_global = 0
    
    base_path = f"/home/jianih/research/EmbodiedBench/running/eb_nav/{exp_name}"
    
    if not os.path.exists(base_path):
        raise ValueError(f"Experiment path does not exist: {base_path}")
    
    # Get all task directories
    task_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d))]
    
    for task in task_dirs:
        task_path = os.path.join(base_path, task)
        
        # Find all episode JSON files
        episode_files = glob.glob(os.path.join(task_path, "episode_*.json"))
        
        for episode_file in episode_files:
            try:
                with open(episode_file, 'r') as f:
                    # Read each line as a separate JSON object
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            step_data = json.loads(line)
                            
                            # Extract scene_graph if it exists
                            if ('reasoning' in step_data and 
                                'scene_graph' in step_data['reasoning']):
                                
                                scene_graph = step_data['reasoning']['scene_graph']
                                total_requests_global += 1
                                
                                # Add to global unique keywords and counts
                                for key in global_unique_keywords.keys():
                                    if key in scene_graph:
                                        values = scene_graph[key]
                                        if isinstance(values, list):
                                            global_keyword_counts[key] += len(values)
                                            # Add to global unique set
                                            for item in values:
                                                if isinstance(item, list):
                                                    global_unique_keywords[key].add(str(item))
                                                else:
                                                    global_unique_keywords[key].add(item)
                                        else:
                                            global_keyword_counts[key] += 1
                                            global_unique_keywords[key].add(values)
                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
                            
            except Exception as e:
                print(f"Warning: Could not process {episode_file}: {e}")
                continue
    
    # Calculate global averages
    global_avg_keywords = {
        "target_name": 0.0,
        "target_attribute": 0.0,
        "target_relation": 0.0,
        "related_objects": 0.0
    }
    
    if total_requests_global > 0:
        for key in global_avg_keywords.keys():
            global_avg_keywords[key] = global_keyword_counts[key] / total_requests_global
    
    # Get global unique counts
    global_unique_counts = {key: len(values) for key, values in global_unique_keywords.items()}
    
    return {
        "global_unique_keywords": global_unique_counts,
        "global_avg_keywords_per_request": global_avg_keywords,
        "total_requests_all_tasks": total_requests_global,
        "per_task_stats": per_task_stats
    }


def extract_keywords(exp_name: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract keywords from episode JSON files for a given experiment.
    
    Args:
        exp_name: The experiment name (e.g., 'gpt-4o_gd_laser_topk_3')
    
    Returns:
        Dictionary with structure:
        {
            <TASK>: {
                "target_name": [list of target names],
                "target_attribute": [list of target attributes], 
                "target_relation": [list of target relations],
                "related_objects": [list of related objects]
            }
        }
    """
    base_path = f"/home/jianih/research/EmbodiedBench/running/eb_nav/{exp_name}"
    
    if not os.path.exists(base_path):
        raise ValueError(f"Experiment path does not exist: {base_path}")
    
    result = {}
    
    # Get all task directories
    task_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d))]
    
    for task in task_dirs:
        task_path = os.path.join(base_path, task)
        
        # Initialize aggregated data for this task
        aggregated_data = {
            "target_name": [],
            "target_attribute": [], 
            "target_relation": [],
            "related_objects": []
        }
        
        # Find all episode JSON files
        episode_files = glob.glob(os.path.join(task_path, "episode_*.json"))
        
        for episode_file in episode_files:
            try:
                with open(episode_file, 'r') as f:
                    # Read each line as a separate JSON object
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            step_data = json.loads(line)
                            
                            # Extract scene_graph if it exists
                            if ('reasoning' in step_data and 
                                'scene_graph' in step_data['reasoning']):
                                
                                scene_graph = step_data['reasoning']['scene_graph']
                                
                                # Aggregate keywords from this step
                                for key in aggregated_data.keys():
                                    if key in scene_graph:
                                        values = scene_graph[key]
                                        if isinstance(values, list):
                                            aggregated_data[key].extend(values)
                                        else:
                                            aggregated_data[key].append(values)
                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
                            
            except Exception as e:
                print(f"Warning: Could not process {episode_file}: {e}")
                continue
        
        # Remove duplicates while preserving order
        for key in aggregated_data.keys():
            # Convert to list of unique items while preserving order
            seen = set()
            unique_items = []
            for item in aggregated_data[key]:
                # Handle nested lists (like target_relation)
                if isinstance(item, list):
                    item_str = str(item)
                    if item_str not in seen:
                        seen.add(item_str)
                        unique_items.append(item)
                else:
                    if item not in seen:
                        seen.add(item)
                        unique_items.append(item)
            aggregated_data[key] = unique_items
        
        result[task] = aggregated_data
    
    return result



if __name__ == "__main__":
    # Example usage
    exp_name = "gpt-4o_laser_combined"
    
    print("=== KEYWORD EXTRACTION ===")
    keywords = extract_keywords(exp_name)
    
    # Pretty print the keyword extraction results
    for task, data in keywords.items():
        print(f"\n--- {task.upper()} ---")
        for key, values in data.items():
            print(f"{key}: {values}")
    
    print("\n\n=== PER-TASK KEYWORD STATISTICS ===")
    stats = calculate_keyword_stats(exp_name)
    
    # Pretty print the per-task statistics
    for task, data in stats.items():
        print(f"\n--- {task.upper()} STATS ---")
        print(f"Total requests: {data['total_requests']}")
        print("Average keywords per request:")
        for key, avg in data['avg_keywords_per_request'].items():
            print(f"  {key}: {avg:.2f}")
        print("Unique keywords total:")
        for key, count in data['unique_keywords_total'].items():
            print(f"  {key}: {count}")
    
    print("\n\n=== GLOBAL KEYWORD STATISTICS (ACROSS ALL TASKS) ===")
    global_stats = calculate_global_keyword_stats(exp_name)
    
    print(f"Total requests across all tasks: {global_stats['total_requests_all_tasks']}")
    print("\nGlobal average keywords per request:")
    for key, avg in global_stats['global_avg_keywords_per_request'].items():
        print(f"  {key}: {avg:.2f}")
    print("\nGlobal unique keywords across all tasks:")
    for key, count in global_stats['global_unique_keywords'].items():
        print(f"  {key}: {count}")
