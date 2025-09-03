import json
import random
import math
import os
import sys
from collections import Counter, defaultdict

def compute_dual_agreement_score(execution_results_a, execution_results_b):
    """
    Compute dual agreement score using clustering approach with two code variants
    """
    def compute_single_score(execution_results):
        if not execution_results:
            return 0, []
        
        # Group execution results by their test case patterns (case sets)
        caseset_to_indices = defaultdict(list)
        for idx, result in enumerate(execution_results):
            # Convert test results to tuple of pass/fail (1/0)
            if 'test_results' in result:
                case_set = tuple(1 if tc == "1" else 0 for tc in result['test_results'])
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
                'total_tests': len(case_set)
            })
        
        # Sort by score descending and return the best score
        caseset_scores.sort(key=lambda x: x['score'], reverse=True)
        best_score = caseset_scores[0]['score'] if caseset_scores else 0
        return best_score, caseset_scores

    # Compute scores for both code variants
    score_a, clusters_a = compute_single_score(execution_results_a)
    score_b, clusters_b = compute_single_score(execution_results_b)
    
    return {
        'variant_a_score': score_a,
        'variant_b_score': score_b,
        'variant_a_clusters': clusters_a,
        'variant_b_clusters': clusters_b
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
    
    model_dir = f'test_result5/{model}_results'
    
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

def process_dual_agreement(technique_a, technique_b, model):
    """Process dual agreement between two techniques using cluster-based scoring"""
    print(f"\n{'='*50}")
    print(f"Processing: {technique_a} vs {technique_b} ({model})")
    print(f"{'='*50}")
    
    # Load cross technique data
    cross_results = load_cross_technique_data(model)
    
    # Load ground truth
    ground_truth = load_ground_truth_results(model)
    print(f"Loaded {len(ground_truth)} ground truth entries")
    
    # Group results by task_id for dual comparison
    task_data = defaultdict(lambda: {technique_a: [], technique_b: []})
    
    # Collect all relevant cross-technique results
    for key, data in cross_results.items():
        code_tech, test_tech = key.split('_vs_')
        
        # Include results where either the code or test technique matches our pair
        if code_tech in [technique_a, technique_b] or test_tech in [technique_a, technique_b]:
            for item in data:
                task_id = item.get('task_id')
                if task_id:
                    # Determine which technique this result belongs to
                    if code_tech == technique_a or test_tech == technique_a:
                        task_data[task_id][technique_a].append(item)
                    elif code_tech == technique_b or test_tech == technique_b:
                        task_data[task_id][technique_b].append(item)
    
    print(f"Found data for {len(task_data)} tasks")
    
    # Calculate dual agreement scores for each task
    evaluation_results = []
    
    for task_id, data in list(task_data.items())[:5]:  # Just process first 5 for testing
        if not data[technique_a] or not data[technique_b]:
            continue
            
        print(f"\nProcessing task_id {task_id}:")
        print(f"  {technique_a}: {len(data[technique_a])} results")
        print(f"  {technique_b}: {len(data[technique_b])} results")
        
        # Compute dual agreement scores
        agreement_results = compute_dual_agreement_score(
            data[technique_a], 
            data[technique_b]
        )
        
        score_a = agreement_results['variant_a_score']
        score_b = agreement_results['variant_b_score']
        
        print(f"  {technique_a} cluster score: {score_a:.3f}")
        print(f"  {technique_b} cluster score: {score_b:.3f}")
        
        # Select best technique based on dual agreement scores
        if score_a > score_b:
            chosen_technique = technique_a
            chosen_score = score_a
            selection_method = 'dual_agreement'
        elif score_b > score_a:
            chosen_technique = technique_b
            chosen_score = score_b
            selection_method = 'dual_agreement'
        else:
            # Tie - use random selection for now
            chosen_technique = random.choice([technique_a, technique_b])
            chosen_score = score_a  # they're equal
            selection_method = 'random'
            print(f"  Dual agreement tie detected for task {task_id}")
        
        # Get actual result from ground truth
        actual_result = ground_truth.get(task_id, 0)
        
        evaluation_results.append({
            'task_id': task_id,
            'technique_a': technique_a,
            'technique_b': technique_b,
            'technique_a_score': score_a,
            'technique_b_score': score_b,
            'chosen_technique': chosen_technique,
            'chosen_score': chosen_score,
            'actual_result': actual_result,
            'selection_method': selection_method
        })
        
        print(f"  Selected: {chosen_technique}, Score: {chosen_score:.3f}, Actual result: {actual_result}")
    
    return evaluation_results

def main():
    model = 'gemini'
    random.seed(42)
    
    # Test with just two techniques
    technique_a = 'active_tense'
    technique_b = 'declarative_to_interrogative'
    
    results = process_dual_agreement(technique_a, technique_b, model)
    
    if results:
        print(f"\n=== SUMMARY ===")
        print(f"Total results: {len(results)}")
        
        # Calculate statistics
        correct_predictions = sum(1 for r in results if r['actual_result'] == 1)
        total_predictions = len(results)
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        technique_a_wins = sum(1 for r in results if r['chosen_technique'] == technique_a)
        technique_b_wins = sum(1 for r in results if r['chosen_technique'] == technique_b)
        
        print(f"Correct predictions (actual_result=1): {correct_predictions}/{total_predictions}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"{technique_a} wins: {technique_a_wins}")
        print(f"{technique_b} wins: {technique_b_wins}")
        
        # Show detailed results
        print(f"\n=== DETAILED RESULTS ===")
        for r in results:
            print(f"Task {r['task_id']}: {r['chosen_technique']} (score: {r['chosen_score']:.2f}, actual: {r['actual_result']})")
    else:
        print("No results found")

if __name__ == "__main__":
    main()