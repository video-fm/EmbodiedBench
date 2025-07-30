import os
import json
import glob
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import statistics
import numpy as np


def detect_outliers(data: List[float], method='iqr', threshold=3.0) -> Tuple[List[int], List[float]]:
    """
    Detect outliers in timing data.
    
    Args:
        data: List of timing values
        method: 'iqr' for interquartile range or 'zscore' for z-score method
        threshold: Threshold for outlier detection
    
    Returns:
        Tuple of (outlier_indices, outlier_values)
    """
    if len(data) < 4:  # Need at least 4 points for meaningful outlier detection
        return [], []
    
    data_array = np.array(data)
    outlier_indices = []
    outlier_values = []
    
    if method == 'iqr':
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + threshold * IQR  # Use custom threshold for upper bound
        
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                outlier_indices.append(i)
                outlier_values.append(value)
    
    elif method == 'zscore':
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        if std_val > 0:
            z_scores = np.abs((data_array - mean_val) / std_val)
            for i, z_score in enumerate(z_scores):
                if z_score > threshold:
                    outlier_indices.append(i)
                    outlier_values.append(data[i])
    
    # Additional check: anything over 5 minutes is likely debugging
    for i, value in enumerate(data):
        if value > 300 and i not in outlier_indices:  # 5 minutes
            outlier_indices.append(i)
            outlier_values.append(value)
    
    return outlier_indices, outlier_values


def analyze_episode_timing(episode_file: str) -> Dict[str, Any]:
    """Analyze timing for a single episode."""
    planning_times = []
    execution_times = []
    planning_details = []  # Store details about each planning instance
    execution_details = []  # Store details about each execution instance
    
    try:
        times = []
        step_numbers = []
        with open(episode_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    step_data = json.loads(line)
                    if 'episode_elapsed_seconds' in step_data and 'env_step' in step_data:
                        times.append(step_data['episode_elapsed_seconds'])
                        step_numbers.append(step_data['env_step'])
                except json.JSONDecodeError:
                    continue
        
        if len(times) < 2:
            return {
                'planning_times': [],
                'execution_times': [],
                'planning_details': [],
                'execution_details': [],
                'total_steps': len(times),
                'total_episode_time': times[-1] if times else 0
            }
        
        # Calculate time differences between consecutive steps
        time_diffs = []
        for i in range(1, len(times)):
            diff = times[i] - times[i-1]
            time_diffs.append(diff)
        
        # Classify time differences as planning or execution
        # Use a threshold-based approach: large gaps are planning, small gaps are execution
        # We'll use the median as a rough threshold, but also consider absolute values
        if time_diffs:
            # Sort differences to understand distribution
            sorted_diffs = sorted(time_diffs)
            median_diff = statistics.median(time_diffs)
            
            # Threshold: if a time difference is > 5 seconds OR significantly larger than median, it's planning
            planning_threshold = max(5.0, median_diff * 3)
            
            for i, diff in enumerate(time_diffs):
                step_from = step_numbers[i] if i < len(step_numbers) else i
                step_to = step_numbers[i+1] if i+1 < len(step_numbers) else i+1
                
                detail = {
                    'duration': diff,
                    'step_from': step_from,
                    'step_to': step_to,
                    'time_from': times[i],
                    'time_to': times[i+1] if i+1 < len(times) else times[i],
                    'episode_file': episode_file
                }
                
                if diff >= planning_threshold:
                    planning_times.append(diff)
                    planning_details.append(detail)
                else:
                    execution_times.append(diff)
                    execution_details.append(detail)
        
        return {
            'planning_times': planning_times,
            'execution_times': execution_times,
            'planning_details': planning_details,
            'execution_details': execution_details,
            'total_steps': len(times),
            'total_episode_time': times[-1] if times else 0,
            'planning_threshold_used': planning_threshold if time_diffs else 0
        }
        
    except Exception as e:
        print(f"Error processing {episode_file}: {e}")
        return {
            'planning_times': [],
            'execution_times': [],
            'planning_details': [],
            'execution_details': [],
            'total_steps': 0,
            'total_episode_time': 0
        }


def analyze_experiment_timing(exp_name: str) -> Dict[str, Any]:
    """Analyze timing across all episodes in an experiment."""
    base_path = f"/home/jianih/research/EmbodiedBench/running/eb_nav/{exp_name}"
    
    if not os.path.exists(base_path):
        raise ValueError(f"Experiment path does not exist: {base_path}")
    
    # Get all task directories
    task_dirs = [d for d in os.listdir(base_path) 
                 if os.path.isdir(os.path.join(base_path, d))]
    
    all_planning_times = []
    all_execution_times = []
    all_planning_details = []
    all_execution_details = []
    task_results = {}
    episode_stats = []
    
    for task in task_dirs:
        task_path = os.path.join(base_path, task)
        episode_files = glob.glob(os.path.join(task_path, "episode_*.json"))
        
        task_planning_times = []
        task_execution_times = []
        
        for episode_file in episode_files:
            episode_name = os.path.basename(episode_file)
            episode_result = analyze_episode_timing(episode_file)
            
            task_planning_times.extend(episode_result['planning_times'])
            task_execution_times.extend(episode_result['execution_times'])
            
            # Add task info to details
            for detail in episode_result['planning_details']:
                detail['task'] = task
                detail['episode'] = episode_name
            for detail in episode_result['execution_details']:
                detail['task'] = task
                detail['episode'] = episode_name
            
            all_planning_details.extend(episode_result['planning_details'])
            all_execution_details.extend(episode_result['execution_details'])
            
            episode_stats.append({
                'task': task,
                'episode': episode_name,
                'planning_count': len(episode_result['planning_times']),
                'execution_count': len(episode_result['execution_times']),
                'avg_planning_time': statistics.mean(episode_result['planning_times']) if episode_result['planning_times'] else 0,
                'avg_execution_time': statistics.mean(episode_result['execution_times']) if episode_result['execution_times'] else 0,
                'total_planning_time': sum(episode_result['planning_times']),
                'total_execution_time': sum(episode_result['execution_times']),
                'total_episode_time': episode_result['total_episode_time'],
                'total_steps': episode_result['total_steps']
            })
        
        all_planning_times.extend(task_planning_times)
        all_execution_times.extend(task_execution_times)
        
        task_results[task] = {
            'avg_planning_time': statistics.mean(task_planning_times) if task_planning_times else 0,
            'avg_execution_time': statistics.mean(task_execution_times) if task_execution_times else 0,
            'total_planning_instances': len(task_planning_times),
            'total_execution_instances': len(task_execution_times),
            'episodes_analyzed': len(episode_files)
        }
    
    # Detect outliers in planning times
    planning_outlier_indices, planning_outlier_values = detect_outliers(all_planning_times, method='iqr', threshold=2.5)
    
    # Get outlier details
    planning_outliers = []
    for idx in planning_outlier_indices:
        if idx < len(all_planning_details):
            outlier_detail = all_planning_details[idx].copy()
            outlier_detail['is_outlier'] = True
            planning_outliers.append(outlier_detail)
    
    # Filter out outliers for clean statistics
    filtered_planning_times = [time for i, time in enumerate(all_planning_times) if i not in planning_outlier_indices]
    
    # Calculate statistics with and without outliers
    def calc_stats(times_list, label):
        if not times_list:
            return {
                f'{label}_avg': 0,
                f'{label}_median': 0,
                f'{label}_std': 0,
                f'{label}_count': 0
            }
        return {
            f'{label}_avg': statistics.mean(times_list),
            f'{label}_median': statistics.median(times_list),
            f'{label}_std': statistics.stdev(times_list) if len(times_list) > 1 else 0,
            f'{label}_count': len(times_list)
        }
    
    # Statistics with outliers
    raw_planning_stats = calc_stats(all_planning_times, 'raw_planning')
    raw_execution_stats = calc_stats(all_execution_times, 'raw_execution')
    
    # Statistics without outliers
    filtered_planning_stats = calc_stats(filtered_planning_times, 'filtered_planning')
    filtered_execution_stats = calc_stats(all_execution_times, 'filtered_execution')  # Execution times typically don't need filtering
    
    global_stats = {
        **raw_planning_stats,
        **raw_execution_stats,
        **filtered_planning_stats,
        **filtered_execution_stats,
        'outliers_detected': len(planning_outliers),
        'outlier_percentage': (len(planning_outliers) / len(all_planning_times) * 100) if all_planning_times else 0
    }
    
    return {
        'global_stats': global_stats,
        'task_stats': task_results,
        'episode_details': episode_stats,
        'planning_outliers': planning_outliers,
        'raw_planning_times': all_planning_times,
        'filtered_planning_times': filtered_planning_times,
        'raw_execution_times': all_execution_times
    }


def print_timing_analysis(exp_name: str):
    """Print a comprehensive timing analysis for an experiment."""
    results = analyze_experiment_timing(exp_name)
    
    print(f"=== TIMING ANALYSIS FOR {exp_name.upper()} ===\n")
    
    # Outlier detection results
    outliers = results['planning_outliers']
    if outliers:
        print("🚨 OUTLIER DETECTION RESULTS:")
        print(f"  Found {len(outliers)} planning outliers ({results['global_stats']['outlier_percentage']:.1f}% of all planning instances)")
        print(f"  These are likely due to debugging, pauses, or system issues:\n")
        
        # Sort outliers by duration (longest first)
        sorted_outliers = sorted(outliers, key=lambda x: x['duration'], reverse=True)
        
        for i, outlier in enumerate(sorted_outliers[:10]):  # Show top 10 outliers
            minutes = int(outlier['duration'] // 60)
            seconds = outlier['duration'] % 60
            print(f"    {i+1:2d}. {outlier['task']}/{outlier['episode']} steps {outlier['step_from']}-{outlier['step_to']}: "
                  f"{minutes:2d}m {seconds:05.2f}s")
        
        if len(sorted_outliers) > 10:
            print(f"    ... and {len(sorted_outliers) - 10} more outliers")
        
        print()
    
    # Global statistics comparison
    global_stats = results['global_stats']
    print("📊 OVERALL STATISTICS:")
    print("  RAW DATA (including outliers):")
    print(f"    Average Planning Time: {global_stats['raw_planning_avg']:.2f} seconds")
    print(f"    Average Execution Time: {global_stats['raw_execution_avg']:.2f} seconds")
    print(f"    Planning Instances: {global_stats['raw_planning_count']}")
    print(f"    Execution Instances: {global_stats['raw_execution_count']}")
    
    print("\n  FILTERED DATA (outliers removed):")
    print(f"    Average Planning Time: {global_stats['filtered_planning_avg']:.2f} seconds")
    print(f"    Average Execution Time: {global_stats['filtered_execution_avg']:.2f} seconds")
    print(f"    Planning Instances: {global_stats['filtered_planning_count']}")
    print(f"    Execution Instances: {global_stats['filtered_execution_count']}")
    
    # Show the difference
    if global_stats['raw_planning_avg'] > 0 and global_stats['filtered_planning_avg'] > 0:
        improvement = ((global_stats['raw_planning_avg'] - global_stats['filtered_planning_avg']) / 
                      global_stats['raw_planning_avg'] * 100)
        print(f"\n  🎯 FILTERING IMPACT:")
        print(f"    Planning time reduced by {improvement:.1f}% after removing outliers")
        print(f"    From {global_stats['raw_planning_avg']:.2f}s to {global_stats['filtered_planning_avg']:.2f}s average")
    
    # Ratios
    if global_stats['filtered_planning_avg'] > 0 and global_stats['filtered_execution_avg'] > 0:
        ratio = global_stats['filtered_planning_avg'] / global_stats['filtered_execution_avg']
        print(f"    Clean Planning/Execution Ratio: {ratio:.2f}x")
    
    # Task-level statistics
    print(f"\n📋 PER-TASK BREAKDOWN:")
    for task, stats in results['task_stats'].items():
        print(f"\n  --- {task.upper()} ---")
        print(f"    Episodes Analyzed: {stats['episodes_analyzed']}")
        print(f"    Avg Planning Time: {stats['avg_planning_time']:.2f}s")
        print(f"    Avg Execution Time: {stats['avg_execution_time']:.2f}s")
        print(f"    Planning Instances: {stats['total_planning_instances']}")
        print(f"    Execution Instances: {stats['total_execution_instances']}")
        
        if stats['avg_planning_time'] > 0 and stats['avg_execution_time'] > 0:
            task_ratio = stats['avg_planning_time'] / stats['avg_execution_time']
            print(f"    Planning/Execution Ratio: {task_ratio:.2f}x")
    
    # Episode details summary
    print(f"\n📈 EPISODE SUMMARY:")
    episodes = results['episode_details']
    if episodes:
        total_episodes = len(episodes)
        avg_episode_time = statistics.mean([ep['total_episode_time'] for ep in episodes])
        avg_steps_per_episode = statistics.mean([ep['total_steps'] for ep in episodes])
        
        print(f"  Total Episodes: {total_episodes}")
        print(f"  Average Episode Duration: {avg_episode_time:.2f} seconds")
        print(f"  Average Steps per Episode: {avg_steps_per_episode:.1f}")
        
        # Planning vs execution time breakdown
        total_planning_time = sum([ep['total_planning_time'] for ep in episodes])
        total_execution_time = sum([ep['total_execution_time'] for ep in episodes])
        total_time = total_planning_time + total_execution_time
        
        if total_time > 0:
            planning_percentage = (total_planning_time / total_time) * 100
            execution_percentage = (total_execution_time / total_time) * 100
            
            print(f"  Time Distribution:")
            print(f"    Planning: {planning_percentage:.1f}% ({total_planning_time:.1f}s total)")
            print(f"    Execution: {execution_percentage:.1f}% ({total_execution_time:.1f}s total)")
    
    return results


if __name__ == "__main__":
    # Example usage
    # exp_name = "gpt-4o_laser_combined"
    exp_name = "gpt-4o_baseline_sg"
    results = print_timing_analysis(exp_name)
    
    # Optionally save detailed results to a file
    output_file = f"timing_analysis_{exp_name}.json"
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'global_stats': results['global_stats'],
            'task_stats': results['task_stats'],
            'episode_details': results['episode_details']
            # Skip raw times to keep file size manageable
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")