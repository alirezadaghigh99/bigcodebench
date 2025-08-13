import json
import random
from collections import Counter, defaultdict
import os

def compute_score(execution_results):
    """
    Compute scores for execution results based on frequency and passing tests.
    
    Args:
        execution_results: List of test result arrays (each array contains 1s and 0s)
    
    Returns:
        List of scores for each execution result
    """
    if not execution_results:
        return []
        
    # Count frequency of each result pattern
    result_counts = Counter(tuple(result) for result in execution_results)
    
    # Calculate score for each result
    scores = []
    for result in execution_results:
        # Count number of passing tests (1s) in the result
        passing_tests = sum(1 for x in result if x == 1)
        # Get frequency of this result pattern
        frequency = result_counts[tuple(result)]
        # Calculate score: frequency * number of passing tests
        score = frequency * passing_tests
        scores.append(score)
    
    return scores

def load_jsonl(file_path):
    """Load JSONL file and return list of JSON objects"""
    try:
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
        return []

def load_ground_truth_results(technique, model):
    """Load ground truth results for a specific technique and model"""
    file_path = f"generation_output/{model}/test_generation/test_code_{technique}_{model}_bigcodebench_output_llm1.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Warning: Ground truth file not found {file_path}")
        return {}
    
    data = load_jsonl(file_path)
    
    return {item['task_id']: item['test_result'] for item in data}

def parse_test_results(test_results_str):
    """Parse test results string to list of integers"""
    if isinstance(test_results_str, list):
        return [int(x) for x in test_results_str]
    elif isinstance(test_results_str, str):
        # Handle string format like "['1', '0', '0', '0']"
        try:
            result_list = eval(test_results_str)
            return [int(x) for x in result_list]
        except:
            return []
    else:
        return []

def load_all_cross_technique_combinations(model=None):
    """Load ALL cross-technique combinations for all techniques"""
    techniques = [
        'original',
        'active_to_passive',
        # 'adversarial_function_name', 
        # 'declarative_to_interrogative',
        # 'lowercase_to_uppercase',
        # 'rephrase_prompt',
        # 'task_function_name',
        # 'verb_to_similar_verb'
    ]
    
    all_combinations = {}
    
    # Load all possible combinations from cross_technique_results
    for code_technique in techniques:
        for test_technique in techniques:
            file_key = f"{code_technique}_test_{test_technique}"
            
            # Try different file patterns based on available models
            file_patterns = [
                f"generation_output/{model}/code_{code_technique}_test_{test_technique}.jsonl",
            ]
            
            # Also try model-specific patterns if model is specified
            if model:
                file_patterns.extend([
                    f"generation_output/{model}/cross_result/code_{code_technique}_{model}_bigcodebench_output_llm1_test_{test_technique}_{model}_bigcodebench_output_llm1.jsonl",
                    f"generation_output/{model}/cross_result/code_{code_technique}_test_{test_technique}.jsonl"
                ])
            
            loaded = False
            for file_path in file_patterns:
                if os.path.exists(file_path):
                    data = load_jsonl(file_path)
                    if data:
                        all_combinations[file_key] = data
                        print(f"Loaded {len(data)} entries for {file_key} from {file_path}")
                        loaded = True
                        break
            
            if not loaded:
                all_combinations[file_key] = []
                print(f"No data found for {file_key}")
    
    return all_combinations, techniques

def process_all_techniques_comprehensive(model):
    """Process all techniques together and calculate scores for each technique on each task"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TECHNIQUE EVALUATION ({model})")
    print(f"{'='*80}")
    
    # Load all cross-technique combinations and technique list
    all_combinations, techniques = load_all_cross_technique_combinations(model)
    
    # Load ground truth results for all techniques
    ground_truth_results = {}
    for technique in techniques:
        ground_truth_results[technique] = load_ground_truth_results(technique, model)
        print(f"Loaded {len(ground_truth_results[technique])} ground truth results for {technique}")
    
    # Group results by task_id and code technique
    task_data = defaultdict(lambda: {technique: [] for technique in techniques})
    
    # Process each combination
    print(f"\nProcessing cross-technique test results...")
    for combo_name, combo_data in all_combinations.items():
        if not combo_data:
            continue
            
        # Parse combination name: code_technique_test_test_technique
        parts = combo_name.split('_test_')
        if len(parts) != 2:
            continue
            
        code_technique = parts[0]
        test_technique = parts[1]
        
        print(f"Processing {code_technique} vs {test_technique}: {len(combo_data)} entries")
        
        for item in combo_data:
            task_id = item['task_id']
            item_code_technique = item['code_technique']
            
            # Extract actual technique name from the full code_technique field
            # e.g., 'active_to_passive_o4-mini_bigcodebench_output_llm1' -> 'active_to_passive'
            actual_technique = item_code_technique.split('_' + model + '_')[0] if '_' + model + '_' in item_code_technique else item_code_technique
            
            # Parse test results
            test_results = parse_test_results(item['test_results'])
            
            if not test_results:
                continue
                
            # Add to the appropriate code technique
            if actual_technique in task_data[task_id]:
                task_data[task_id][actual_technique].append(test_results)
    
    print(f"\nFound data for {len(task_data)} tasks")
    # Calculate scores and select best technique for each task
    evaluation_results = []
    
    for task_id, data in task_data.items():
        print(f"\nProcessing task_id {task_id}:")
        
        # Calculate scores for each technique
        technique_scores = {}
        technique_details = {}
        
        for technique in techniques:
            if data[technique]:
                scores = compute_score(data[technique])
                total_score = sum(scores) if scores else 0
                technique_scores[technique] = total_score
                technique_details[technique] = {
                    'total_score': total_score,
                    'num_executions': len(data[technique]),
                    'individual_scores': scores
                }
                print(f"  {technique}: score={total_score}, executions={len(data[technique])}")
            else:
                technique_scores[technique] = 0
                technique_details[technique] = {
                    'total_score': 0,
                    'num_executions': 0,
                    'individual_scores': []
                }
                print(f"  {technique}: score=0 (no data)")
        
        # Find the technique(s) with the highest score
        max_score = max(technique_scores.values()) if technique_scores else 0
        best_techniques = [tech for tech, score in technique_scores.items() if score == max_score]
        
        # If all techniques have equal scores (all tied), select original; otherwise randomly select one if multiple techniques have same score
        if len(best_techniques) == len(techniques):
            chosen_technique = 'original'
        else:
            chosen_technique = random.choice(best_techniques) if best_techniques else 'original'
        chosen_score = technique_scores[chosen_technique]
        
        # Get actual result for the chosen technique
        actual_result = ground_truth_results[chosen_technique].get(task_id, 0)
        
        evaluation_results.append({
            'task_id': task_id,
            'chosen_solution': chosen_technique,
            'chosen_score': chosen_score,
            'actual_result': actual_result,
            'all_scores': technique_scores,
            'technique_details': technique_details,
            'num_tied_techniques': len(best_techniques),
            'tied_techniques': best_techniques
        })
        
        print(f"  → Selected: {chosen_technique} (score: {chosen_score}, actual: {actual_result})")
        if len(best_techniques) > 1:
            print(f"  → Note: {len(best_techniques)} techniques tied: {best_techniques}")
    
    return evaluation_results

def generate_comprehensive_statistics(results, model):
    """Generate comprehensive statistics from evaluation results"""
    if not results:
        return {}
    
    total_predictions = len(results)
    correct_predictions = sum(1 for r in results if r['actual_result'] == 1)
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Count technique selections
    technique_counts = defaultdict(int)
    technique_correct = defaultdict(int)
    
    for r in results:
        technique = r['chosen_solution']
        technique_counts[technique] += 1
        if r['actual_result'] == 1:
            technique_correct[technique] += 1
    
    # Count ties
    total_ties = sum(1 for r in results if r['num_tied_techniques'] > 1)
    tie_distribution = defaultdict(int)
    for r in results:
        tie_distribution[r['num_tied_techniques']] += 1
    
    # Score distribution analysis
    score_stats = {
        'min_score': min(r['chosen_score'] for r in results),
        'max_score': max(r['chosen_score'] for r in results),
        'avg_score': sum(r['chosen_score'] for r in results) / total_predictions,
        'zero_score_tasks': sum(1 for r in results if r['chosen_score'] == 0)
    }
    
    # Technique-specific accuracy
    technique_accuracy = {}
    for technique in technique_counts:
        if technique_counts[technique] > 0:
            technique_accuracy[technique] = (technique_correct[technique] / technique_counts[technique]) * 100
        else:
            technique_accuracy[technique] = 0
    
    return {
        'model': model,
        'total_tasks': total_predictions,
        'correct_predictions': correct_predictions,
        'overall_accuracy': accuracy,
        'technique_selection_counts': dict(technique_counts),
        'technique_correct_counts': dict(technique_correct),
        'technique_accuracy': technique_accuracy,
        'tie_statistics': {
            'total_ties': total_ties,
            'tie_distribution': dict(tie_distribution)
        },
        'score_statistics': score_stats
    }

def main():
    import sys
    
    # Get model from command line argument or default to o4-mini
    model = sys.argv[1] if len(sys.argv) > 1 else 'o4-mini'
    
    random.seed(42)  # For reproducible results
    
    # Process all techniques together
    print(f"Starting comprehensive evaluation for model: {model}")
    results = process_all_techniques_comprehensive(model)
    
    if results:
        # Create directory structure
        output_dir = f'comprehensive_evaluation_results1/{model}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        output_file = f'{output_dir}/comprehensive_technique_evaluation.jsonl'
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"\nSaved {len(results)} detailed results to {output_file}")
        
        # Generate and save comprehensive statistics
        stats = generate_comprehensive_statistics(results, model)
        
        stats_file = f'{output_dir}/comprehensive_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved comprehensive statistics to {stats_file}")
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION SUMMARY ({model})")
        print(f"{'='*80}")
        print(f"Total tasks evaluated: {stats['total_tasks']}")
        print(f"Correct predictions: {stats['correct_predictions']}")
        print(f"Overall accuracy: {stats['overall_accuracy']:.2f}%")
        print(f"Tasks with tied scores: {stats['tie_statistics']['total_ties']}")
        print(f"Tasks with zero scores: {stats['score_statistics']['zero_score_tasks']}")
        
        print(f"\nTechnique Selection Summary:")
        for technique, count in sorted(stats['technique_selection_counts'].items()):
            percentage = (count / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0
            accuracy = stats['technique_accuracy'][technique]
            correct = stats['technique_correct_counts'][technique]
            print(f"  {technique}: {count} selections ({percentage:.1f}%), {correct} correct ({accuracy:.1f}% accuracy)")
        
        print(f"\nTie Distribution:")
        for num_tied, count in sorted(stats['tie_statistics']['tie_distribution'].items()):
            percentage = (count / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0
            print(f"  {num_tied} techniques tied: {count} tasks ({percentage:.1f}%)")
        
        print(f"\nScore Statistics:")
        print(f"  Min score: {stats['score_statistics']['min_score']}")
        print(f"  Max score: {stats['score_statistics']['max_score']}")
        print(f"  Average score: {stats['score_statistics']['avg_score']:.2f}")
        
    else:
        print("No results found - check data availability")

if __name__ == "__main__":
    main()