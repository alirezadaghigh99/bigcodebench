import json
import random
import math
import os
import sys
from collections import Counter, defaultdict
from code_quality_analyzer import compare_code_quality_detailed
from itertools import combinations

def compute_multi_technique_clustering_score(execution_results_dict):
    """
    Compute clustering scores for multiple techniques simultaneously.
    Groups execution results by test case patterns and calculates scores for each technique
    using the CodeT clustering formula: case_set_score × sqrt(solution_set_size)
    
    Args:
        execution_results_dict: Dict[technique_name, List[execution_results]]
    
    Returns:
        Dict with technique scores and detailed clustering information
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
    
    # Compute scores for all techniques
    technique_scores = {}
    all_clusters = {}
    
    for technique_name, execution_results in execution_results_dict.items():
        score, clusters = compute_single_technique_score(execution_results, technique_name)
        technique_scores[technique_name] = score
        all_clusters[technique_name] = clusters
    
    # Find the best technique
    best_technique = max(technique_scores.keys(), key=lambda t: technique_scores[t]) if technique_scores else None
    best_score = technique_scores.get(best_technique, 0) if best_technique else 0
    
    # Identify ties (techniques with same best score)
    ties = [t for t, score in technique_scores.items() if score == best_score] if best_score > 0 else []
    
    return {
        'technique_scores': technique_scores,
        'best_technique': best_technique,
        'best_score': best_score,
        'tied_techniques': ties,
        'has_tie': len(ties) > 1,
        'all_clusters': all_clusters,
        'total_techniques_evaluated': len(technique_scores)
    }

def load_jsonl(file_path):
    """Load JSONL file and return list of JSON objects"""
    try:
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
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
                    # Handle different task_id field names
                    task_id = item.get('task_id') or item.get('id')
                    if task_id:
                        # Extract test_result, handling different formats
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
            # Format: code_{technique1}_{model}_test_{technique2}_{model}.jsonl
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
    
    return list(techniques)

def resolve_ties_with_quality_analysis(tied_techniques, task_id, model):
    """Resolve ties between multiple techniques using code quality analysis"""
    if len(tied_techniques) <= 1:
        return tied_techniques[0] if tied_techniques else None, None
    
    print(f"    Resolving {len(tied_techniques)}-way tie using quality analysis: {tied_techniques}")
    
    # Perform pairwise quality comparisons between all tied techniques
    quality_scores = defaultdict(int)  # Wins counter for each technique
    quality_details = {}
    
    for tech1, tech2 in combinations(tied_techniques, 2):
        try:
            quality_analysis = compare_code_quality_detailed(task_id, tech1, tech2, model)
            winner = quality_analysis.get('winner')
            
            if winner == tech1:
                quality_scores[tech1] += 1
            elif winner == tech2:
                quality_scores[tech2] += 1
            # If tie or error, no points awarded
            
            quality_details[f"{tech1}_vs_{tech2}"] = quality_analysis
            
        except Exception as e:
            print(f"      Quality analysis error for {tech1} vs {tech2}: {e}")
            continue
    
    # Find technique with most quality wins
    if quality_scores:
        best_quality_technique = max(quality_scores.keys(), key=lambda t: quality_scores[t])
        max_wins = quality_scores[best_quality_technique]
        
        # Check if there's still a tie in quality scores
        quality_tied = [t for t, wins in quality_scores.items() if wins == max_wins]
        
        if len(quality_tied) == 1:
            print(f"      Quality winner: {best_quality_technique} (wins: {max_wins})")
            return best_quality_technique, {
                'method': 'quality_analysis',
                'quality_scores': dict(quality_scores),
                'quality_details': quality_details
            }
    
    # If still tied or quality analysis failed, use random selection
    chosen = random.choice(tied_techniques)
    print(f"      No clear quality winner, randomly selected: {chosen}")
    return chosen, {
        'method': 'random_selection',
        'quality_scores': dict(quality_scores) if quality_scores else {},
        'quality_details': quality_details,
        'reason': 'quality_tie_or_error'
    }

def process_multi_technique_selection(model, max_tasks=None):
    """Process multi-technique selection using cluster-based scoring"""
    print(f"\n{'='*80}")
    print(f"MULTI-TECHNIQUE CLUSTER EVALUATION: ALL TECHNIQUES ({model})")
    print(f"{'='*80}")
    
    # Load cross technique data
    cross_results = load_cross_technique_data(model)
    
    if not cross_results:
        print("No cross-technique data found!")
        return []
    
    # Extract available techniques
    available_techniques = extract_available_techniques(cross_results)
    print(f"Available techniques: {len(available_techniques)}")
    print(f"Techniques: {', '.join(sorted(available_techniques))}")
    
    # Load ground truth
    ground_truth = load_ground_truth_results(model)
    print(f"Ground truth entries: {len(ground_truth)}")
    
    # Group results by task_id and technique combination
    task_technique_data = defaultdict(lambda: defaultdict(list))
    
    # Collect all cross-technique combinations for each task
    for key, data in cross_results.items():
        code_tech, test_tech = key.split('_vs_')
        
        for item in data:
            task_id = item.get('task_id')
            if task_id:
                # Create a comprehensive key for this specific combination
                combo_key = f"{code_tech}_code_{test_tech}_test"
                task_technique_data[task_id][combo_key].append(item)
    
    print(f"Tasks with cross-technique data: {len(task_technique_data)}")
    
    # Process each task for multi-technique selection
    evaluation_results = []
    processed_count = 0
    
    for task_id, technique_combinations in task_technique_data.items():
        if max_tasks and processed_count >= max_tasks:
            break
            
        print(f"\nProcessing task_id {task_id}:")
        print(f"  Found {len(technique_combinations)} technique combinations")
        
        # Group execution results by technique (combining all test combinations for each technique)
        technique_execution_data = defaultdict(list)
        
        for combo_key, combo_results in technique_combinations.items():
            # Extract technique name from combination key
            if '_code_' in combo_key:
                technique_name = combo_key.split('_code_')[0]
                technique_execution_data[technique_name].extend(combo_results)
        
        if len(technique_execution_data) < 2:
            print(f"  Insufficient techniques for comparison ({len(technique_execution_data)}), skipping")
            continue
        
        print(f"  Comparing {len(technique_execution_data)} techniques: {list(technique_execution_data.keys())}")
        
        # Compute multi-technique clustering scores
        clustering_results = compute_multi_technique_clustering_score(technique_execution_data)
        
        technique_scores = clustering_results['technique_scores']
        best_technique = clustering_results['best_technique']
        tied_techniques = clustering_results['tied_techniques']
        has_tie = clustering_results['has_tie']
        
        print(f"  Technique scores: {dict(sorted(technique_scores.items(), key=lambda x: x[1], reverse=True))}")
        print(f"  Best technique: {best_technique} (score: {clustering_results['best_score']:.3f})")
        
        # Handle ties
        tie_resolution = None
        if has_tie and len(tied_techniques) > 1:
            print(f"  Tie detected between: {tied_techniques}")
            chosen_technique, tie_resolution = resolve_ties_with_quality_analysis(tied_techniques, task_id, model)
            selection_method = 'clustering_with_tie_resolution'
        else:
            chosen_technique = best_technique
            selection_method = 'clustering'
        
        # Get actual result from ground truth
        actual_result = ground_truth.get(task_id, 0)
        
        result_entry = {
            'task_id': task_id,
            'available_techniques': list(technique_scores.keys()),
            'technique_scores': technique_scores,
            'chosen_technique': chosen_technique,
            'chosen_score': technique_scores.get(chosen_technique, 0),
            'actual_result': actual_result,
            'selection_method': selection_method,
            'had_tie': has_tie,
            'tied_techniques': tied_techniques if has_tie else [],
            'tie_resolution': tie_resolution,
            'clustering_details': clustering_results,
            'technique_combination_counts': {k: len(v) for k, v in technique_combinations.items()}
        }
        
        evaluation_results.append(result_entry)
        processed_count += 1
        
        print(f"  Selected: {chosen_technique}, Score: {technique_scores.get(chosen_technique, 0):.3f}, Actual result: {actual_result}")
    
    return evaluation_results

def main():
    # Get model from command line argument or default to gemini
    model = sys.argv[1] if len(sys.argv) > 1 else 'gemini'
    # Optional: limit number of tasks for testing
    max_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    random.seed(42)  # For reproducible results
    
    # Create output directory
    output_dir = f'multi_technique_results/{model}'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        results = process_multi_technique_selection(model, max_tasks)
        
        if results:
            # Save comprehensive results
            output_file = f'{output_dir}/multi_technique_cluster_selection.jsonl'
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            print(f"\nSaved {len(results)} results to {output_file}")
            
            # Calculate statistics
            correct_predictions = sum(1 for r in results if r['actual_result'] == 1)
            total_predictions = len(results)
            accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            # Technique selection statistics
            technique_wins = defaultdict(int)
            tie_resolutions = defaultdict(int)
            selection_methods = defaultdict(int)
            
            for result in results:
                technique_wins[result['chosen_technique']] += 1
                selection_methods[result['selection_method']] += 1
                
                if result.get('tie_resolution'):
                    tie_resolutions[result['tie_resolution']['method']] += 1
            
            print(f"\n{'='*60}")
            print(f"MULTI-TECHNIQUE EVALUATION SUMMARY ({model})")
            print(f"{'='*60}")
            print(f"Overall accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
            print(f"")
            print(f"Technique selection frequency:")
            for technique, count in sorted(technique_wins.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
                print(f"  {technique:25}: {count:3d} ({percentage:5.1f}%)")
            
            print(f"\nSelection methods:")
            for method, count in selection_methods.items():
                percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
                print(f"  {method:30}: {count:3d} ({percentage:5.1f}%)")
            
            if tie_resolutions:
                print(f"\nTie resolution methods:")
                for method, count in tie_resolutions.items():
                    print(f"  {method:30}: {count:3d}")
            
            # Show some examples
            print(f"\n=== SELECTION EXAMPLES ===")
            for i, result in enumerate(results[:5]):
                techniques = result['available_techniques']
                chosen = result['chosen_technique']
                actual = result['actual_result']
                method = result['selection_method']
                print(f"Task {result['task_id']}: {len(techniques)} techniques → chose {chosen} ({method}) → actual: {actual}")
        else:
            print(f"No results found for model {model}")
            
    except Exception as e:
        print(f"Error processing model {model}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()