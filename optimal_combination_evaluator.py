import json
import random
import math
import os
import sys
from collections import Counter, defaultdict
from code_quality_analyzer import compare_code_quality_detailed
from itertools import combinations, chain
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def compute_combination_clustering_score(execution_results_dict, combination_techniques):
    """
    Compute clustering scores for a specific combination of techniques.
    Only considers the specified techniques in the combination.
    
    Args:
        execution_results_dict: Dict[technique_name, List[execution_results]]
        combination_techniques: List[str] - techniques to include in this combination
    
    Returns:
        Dict with combination score and selection details
    """
    def compute_single_technique_score(execution_results, technique_name):
        if not execution_results:
            return 0, []
        
        # Group execution results by their test case patterns (case sets)
        caseset_to_indices = defaultdict(list)
        for idx, result in enumerate(execution_results):
            # Convert test results to tuple of pass/fail (1/0)
            if 'test_results' in result:
                case_set = tuple(1 if tc == "1" else 0 for tc in result['test_results'])
            elif 'test_result' in result:
                # Single test result format
                case_set = (result['test_result'],)
            else:
                # Fallback for different data formats
                case_set = (1 if result.get('pass_rate', 0) > 0 else 0,)
            caseset_to_indices[case_set].append(idx)
        
        # Calculate scores for each case set using CodeT formula
        caseset_scores = []
        for case_set, indices in caseset_to_indices.items():
            case_set_score = sum(case_set)  # number of passing tests
            solution_set_score = math.sqrt(len(indices))  # sqrt of cluster size
            total_score = case_set_score * solution_set_score
            
            caseset_scores.append({
                'case_set': case_set,
                'score': total_score,
                'cluster_size': len(indices),
                'passing_tests': case_set_score,
                'total_tests': len(case_set),
                'technique': technique_name
            })
        
        # Sort by score descending and return the best score
        caseset_scores.sort(key=lambda x: x['score'], reverse=True)
        best_score = caseset_scores[0]['score'] if caseset_scores else 0
        return best_score, caseset_scores
    
    # Compute scores only for techniques in the combination
    technique_scores = {}
    all_clusters = {}
    
    for technique_name in combination_techniques:
        if technique_name in execution_results_dict:
            execution_results = execution_results_dict[technique_name]
            score, clusters = compute_single_technique_score(execution_results, technique_name)
            technique_scores[technique_name] = score
            all_clusters[technique_name] = clusters
        else:
            # Technique not available for this task
            technique_scores[technique_name] = 0
            all_clusters[technique_name] = []
    
    # Find the best technique within this combination
    if technique_scores:
        best_technique = max(technique_scores.keys(), key=lambda t: technique_scores[t])
        best_score = technique_scores[best_technique]
    else:
        best_technique = None
        best_score = 0
    
    # Identify ties (techniques with same best score)
    ties = [t for t, score in technique_scores.items() if score == best_score] if best_score > 0 else []
    
    return {
        'combination': combination_techniques,
        'technique_scores': technique_scores,
        'best_technique': best_technique,
        'best_score': best_score,
        'tied_techniques': ties,
        'has_tie': len(ties) > 1,
        'all_clusters': all_clusters,
        'combination_size': len(combination_techniques)
    }

def resolve_combination_ties(tied_techniques, task_id, model):
    """Resolve ties within a combination using quality analysis"""
    if len(tied_techniques) <= 1:
        return tied_techniques[0] if tied_techniques else None, None
    
    # Use simplified quality resolution for combinations
    quality_scores = defaultdict(int)
    
    for tech1, tech2 in combinations(tied_techniques, 2):
        try:
            quality_analysis = compare_code_quality_detailed(task_id, tech1, tech2, model)
            winner = quality_analysis.get('winner')
            
            if winner == tech1:
                quality_scores[tech1] += 1
            elif winner == tech2:
                quality_scores[tech2] += 1
                
        except Exception:
            continue
    
    if quality_scores:
        best_quality_technique = max(quality_scores.keys(), key=lambda t: quality_scores[t])
        return best_quality_technique, {
            'method': 'quality_analysis',
            'quality_scores': dict(quality_scores)
        }
    
    # Random selection if quality analysis fails
    chosen = random.choice(tied_techniques)
    return chosen, {'method': 'random_selection'}

def load_jsonl(file_path):
    """Load JSONL file and return list of JSON objects"""
    try:
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_ground_truth_results(model):
    """Load ground truth results for a specific model"""
    ground_truth_data = {}
    
    # Load from model-specific directories
    model_dirs = {
        'deepseek': 'test_result5/deepseek_results',
        'gemini': 'test_result5/gemini_results', 
        'o4-mini': 'test_result5/o4-mini_results'
    }
    
    model_dir = model_dirs.get(model, f'test_result5/{model}_results')
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(model_dir, filename)
                data = load_jsonl(file_path)
                for item in data:
                    task_id = item.get('task_id') or item.get('id')
                    if task_id:
                        if 'test_result' in item:
                            ground_truth_data[task_id] = item['test_result']
                        elif 'pass_rate' in item:
                            ground_truth_data[task_id] = 1 if item['pass_rate'] > 0 else 0
                        else:
                            ground_truth_data[task_id] = 0
    
    return ground_truth_data

def load_cross_technique_data(model):
    """Load all cross-technique test results for a model"""
    cross_results = {}
    cross_dir = f'{model}_cross_results'
    
    if not os.path.exists(cross_dir):
        print(f"Warning: Cross results directory not found: {cross_dir}")
        return cross_results
    
    for filename in os.listdir(cross_dir):
        if filename.endswith('.jsonl'):
            # Parse filename to extract code and test techniques
            parts = filename.replace('.jsonl', '').split('_test_')
            if len(parts) == 2:
                code_part = parts[0].replace('code_', '')
                test_part = parts[1]
                
                # Extract technique names (remove model suffixes)
                code_technique = code_part.replace(f'_{model}_bigcodebench_output_llm1_v2', '')
                test_technique = test_part.replace(f'_{model}_bigcodebench_output_llm1_v2', '')
                
                file_path = os.path.join(cross_dir, filename)
                data = load_jsonl(file_path)
                
                key = f"{code_technique}_vs_{test_technique}"
                cross_results[key] = data
    
    return cross_results

def extract_available_techniques(cross_results):
    """Extract all unique techniques from cross results"""
    techniques = set()
    
    for key in cross_results.keys():
        code_tech, test_tech = key.split('_vs_')
        techniques.add(code_tech)
        techniques.add(test_tech)
    
    return sorted(list(techniques))

def evaluate_single_task_combinations(task_data, available_techniques, combination_sizes, task_id, model, ground_truth):
    """Evaluate all combinations for a single task"""
    task_id, technique_combinations = task_data
    
    # Group execution results by technique
    technique_execution_data = defaultdict(list)
    
    for combo_key, combo_results in technique_combinations.items():
        if '_code_' in combo_key:
            technique_name = combo_key.split('_code_')[0]
            technique_execution_data[technique_name].extend(combo_results)
    
    if len(technique_execution_data) < 2:
        return None
    
    # Generate all combinations of specified sizes
    all_combinations = []
    for size in combination_sizes:
        if size <= len(available_techniques):
            size_combinations = list(combinations(available_techniques, size))
            all_combinations.extend(size_combinations)
    
    # Evaluate each combination
    combination_results = []
    
    for combination in all_combinations:
        # Filter to only techniques available for this task
        available_in_combination = [t for t in combination if t in technique_execution_data]
        
        if len(available_in_combination) >= 2:  # Need at least 2 techniques
            clustering_result = compute_combination_clustering_score(
                technique_execution_data, 
                list(combination)
            )
            
            # Handle ties within combination
            chosen_technique = clustering_result['best_technique']
            tie_resolution = None
            
            if clustering_result['has_tie']:
                chosen_technique, tie_resolution = resolve_combination_ties(
                    clustering_result['tied_techniques'], task_id, model
                )
            
            # Get actual result
            actual_result = ground_truth.get(task_id, 0)
            
            combination_results.append({
                'task_id': task_id,
                'combination': list(combination),
                'combination_size': len(combination),
                'available_techniques_in_combination': available_in_combination,
                'technique_scores': clustering_result['technique_scores'],
                'chosen_technique': chosen_technique,
                'chosen_score': clustering_result['technique_scores'].get(chosen_technique, 0),
                'actual_result': actual_result,
                'had_tie': clustering_result['has_tie'],
                'tie_resolution': tie_resolution,
                'clustering_details': clustering_result
            })
    
    return combination_results

def process_combination_evaluation(model, combination_sizes=None, max_tasks=None):
    """Process optimal combination evaluation using cluster-based scoring"""
    print(f"\n{'='*80}")
    print(f"OPTIMAL COMBINATION EVALUATION: FINDING BEST TECHNIQUE COMBINATIONS ({model})")
    print(f"{'='*80}")
    
    if combination_sizes is None:
        combination_sizes = [2, 3, 4, 5]  # Default sizes to evaluate
    
    # Load cross technique data
    cross_results = load_cross_technique_data(model)
    
    if not cross_results:
        print("No cross-technique data found!")
        return []
    
    # Extract available techniques
    available_techniques = extract_available_techniques(cross_results)
    print(f"Available techniques: {len(available_techniques)}")
    print(f"Techniques: {', '.join(available_techniques)}")
    print(f"Evaluating combination sizes: {combination_sizes}")
    
    # Calculate total combinations to evaluate
    total_combinations = sum(math.comb(len(available_techniques), size) 
                           for size in combination_sizes 
                           if size <= len(available_techniques))
    print(f"Total combinations to evaluate per task: {total_combinations}")
    
    # Load ground truth
    ground_truth = load_ground_truth_results(model)
    print(f"Ground truth entries: {len(ground_truth)}")
    
    # Group results by task_id and technique combination
    task_technique_data = defaultdict(lambda: defaultdict(list))
    
    for key, data in cross_results.items():
        code_tech, test_tech = key.split('_vs_')
        
        for item in data:
            task_id = item.get('task_id')
            if task_id:
                combo_key = f"{code_tech}_code_{test_tech}_test"
                task_technique_data[task_id][combo_key].append(item)
    
    print(f"Tasks with cross-technique data: {len(task_technique_data)}")
    
    # Limit tasks if specified
    task_list = list(task_technique_data.items())
    if max_tasks:
        task_list = task_list[:max_tasks]
        print(f"Limited to {len(task_list)} tasks for evaluation")
    
    # Evaluate combinations for each task
    print(f"\nEvaluating combinations...")
    all_combination_results = []
    
    start_time = time.time()
    
    for i, (task_id, technique_combinations) in enumerate(task_list):
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (i + 1)) * (len(task_list) - i - 1)
            print(f"  Progress: {i+1}/{len(task_list)} tasks ({(i+1)/len(task_list)*100:.1f}%) - ETA: {remaining/60:.1f}min")
        
        task_results = evaluate_single_task_combinations(
            (task_id, technique_combinations),
            available_techniques,
            combination_sizes,
            task_id,
            model,
            ground_truth
        )
        
        if task_results:
            all_combination_results.extend(task_results)
    
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed/60:.1f} minutes")
    
    return all_combination_results

def analyze_combination_performance(results):
    """Analyze and rank combination performance"""
    if not results:
        return {}
    
    # Group results by combination
    combination_performance = defaultdict(list)
    
    for result in results:
        combination_key = tuple(sorted(result['combination']))
        combination_performance[combination_key].append(result)
    
    # Calculate performance metrics for each combination
    combination_stats = {}
    
    for combination, task_results in combination_performance.items():
        correct = sum(1 for r in task_results if r['actual_result'] == 1)
        total = len(task_results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Technique selection frequency within this combination
        technique_wins = defaultdict(int)
        for r in task_results:
            if r['chosen_technique']:
                technique_wins[r['chosen_technique']] += 1
        
        combination_stats[combination] = {
            'combination': list(combination),
            'combination_size': len(combination),
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'technique_wins': dict(technique_wins),
            'most_selected_technique': max(technique_wins.items(), key=lambda x: x[1])[0] if technique_wins else None
        }
    
    return combination_stats

def main():
    # Parse command line arguments
    model = sys.argv[1] if len(sys.argv) > 1 else 'gemini'
    max_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Combination sizes to evaluate
    combination_sizes = [2, 3, 4]  # Start with smaller combinations
    if len(sys.argv) > 3:
        # Allow custom combination sizes
        combination_sizes = [int(x) for x in sys.argv[3].split(',')]
    
    random.seed(42)  # For reproducible results
    
    # Create output directory
    output_dir = f'combination_results/{model}'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Starting combination evaluation for {model}...")
        results = process_combination_evaluation(model, combination_sizes, max_tasks)
        
        if results:
            # Save detailed results
            output_file = f'{output_dir}/combination_evaluation_results.jsonl'
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            print(f"\nSaved {len(results)} detailed results to {output_file}")
            
            # Analyze combination performance
            combination_stats = analyze_combination_performance(results)
            
            # Save combination analysis
            stats_file = f'{output_dir}/combination_performance_analysis.json'
            with open(stats_file, 'w') as f:
                # Convert tuple keys to strings for JSON serialization
                json_stats = {str(k): v for k, v in combination_stats.items()}
                json.dump(json_stats, f, indent=2)
            
            print(f"Saved combination analysis to {stats_file}")
            
            # Print top combinations by accuracy
            print(f"\n{'='*80}")
            print(f"TOP PERFORMING COMBINATIONS ({model})")
            print(f"{'='*80}")
            
            # Sort by accuracy
            sorted_combinations = sorted(combination_stats.items(), 
                                       key=lambda x: x[1]['accuracy'], 
                                       reverse=True)
            
            for i, (combination, stats) in enumerate(sorted_combinations[:10]):
                print(f"\n{i+1}. Combination (size {stats['combination_size']}): {stats['combination']}")
                print(f"   Accuracy: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
                print(f"   Most selected: {stats['most_selected_technique']}")
                print(f"   Technique wins: {stats['technique_wins']}")
            
            # Analysis by combination size
            print(f"\n{'='*60}")
            print("PERFORMANCE BY COMBINATION SIZE")
            print(f"{'='*60}")
            
            size_performance = defaultdict(list)
            for stats in combination_stats.values():
                size_performance[stats['combination_size']].append(stats['accuracy'])
            
            for size in sorted(size_performance.keys()):
                accuracies = size_performance[size]
                avg_accuracy = sum(accuracies) / len(accuracies)
                max_accuracy = max(accuracies)
                min_accuracy = min(accuracies)
                
                print(f"Size {size}: {len(accuracies)} combinations")
                print(f"  Avg accuracy: {avg_accuracy:.1f}%")
                print(f"  Best: {max_accuracy:.1f}%, Worst: {min_accuracy:.1f}%")
        else:
            print(f"No results found for model {model}")
            
    except Exception as e:
        print(f"Error processing model {model}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()