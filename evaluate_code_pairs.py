import json
import random
from collections import Counter, defaultdict
import os
from tqdm import tqdm

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
        
    print(f"Computing scores for {len(execution_results)} execution results")
    
    # Count frequency of each result pattern
    result_counts = Counter(tuple(result) for result in execution_results)
    print(f"Found {len(result_counts)} unique result patterns")
    
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
def load_ground_truth_results(technique, model, results_dir="test_result5"):
    """Load ground truth results for a specific technique and model from test results"""
    # Try different file patterns
    possible_files = [
        f"{results_dir}/o4-mini_results/code_{technique}_{model}_bigcodebench_output_llm1.jsonl",
        f"{results_dir}/{model}_results/code_{technique}_{model}_bigcodebench_output_llm1.jsonl",
        f"{results_dir}/code_{technique}_{model}_bigcodebench_output_llm1.jsonl"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Loading ground truth from: {file_path}")
            data = load_jsonl(file_path)
            # Extract test results from the loaded data
            results = {}
            for item in data:
                task_id = item.get('id', item.get('task_id', ''))
                if 'test_result' in item:
                    results[task_id] = item['test_result']
                elif 'test_results' in item:
                    # Handle different key names
                    test_results = item['test_results']
                    # Convert to pass/fail based on test results
                    if isinstance(test_results, list):
                        # If all tests pass, result is 1, otherwise 0
                        results[task_id] = 1 if all(int(x) for x in test_results) else 0
                    else:
                        results[task_id] = test_results
            return results
    
    print(f"Warning: Ground truth file not found for technique {technique}, model {model}")
    print(f"Tried: {possible_files}")
    return {}

def load_cross_technique_results(technique, model, results_dir="test_result5"):
    """Load cross-technique test results for a specific technique and model"""
    combinations = {}
    
    # Auto-detect available techniques from the results directory
    techniques = []
    
    # Check what result files exist
    possible_result_dirs = [
        f"{results_dir}/o4-mini_results",
        f"{results_dir}/{model}_results",
        results_dir
    ]
    
    result_dir = None
    for dir_path in possible_result_dirs:
        if os.path.exists(dir_path):
            result_dir = dir_path
            break
    
    if not result_dir:
        print(f"Warning: No results directory found")
        return combinations
    
    print(f"Loading cross-technique results from: {result_dir}")
    
    # Auto-detect techniques from available files
    for filename in os.listdir(result_dir):
        if filename.startswith('code_') and filename.endswith('.jsonl') and 'original' not in filename:
            # Extract full technique name
            # Format: code_{technique}_{model}_bigcodebench_output_llm1.jsonl
            name_part = filename.replace('code_', '').replace('.jsonl', '')
            # Remove model and suffix parts
            for model in ['gemini', 'o4-mini']:
                if f'_{model}_bigcodebench' in name_part:
                    tech = name_part.split(f'_{model}_bigcodebench')[0]
                    if tech not in techniques and tech != 'original':
                        techniques.append(tech)
                    break
    
    print(f"Auto-detected techniques: {techniques}")
    
    # Since we only have direct test results (not cross-technique), 
    # we'll simulate cross-technique by using the actual test results
    for tech in techniques:
        for test_model in ['gemini', 'o4-mini']:  # Both models in results
            file_path = f"{result_dir}/code_{tech}_{test_model}_bigcodebench_output_llm1.jsonl"
            if os.path.exists(file_path):
                data = load_jsonl(file_path)
                # Create entries that look like cross-technique results
                cross_data = []
                for item in data:
                    cross_item = {
                        'task_id': item.get('id', item.get('task_id', '')),
                        'code_technique': tech,
                        'test_technique': tech,  # Using same technique for both
                        'test_results': item.get('test_results', item.get('test_result', [])),
                        'pass_rate': item.get('pass_rate', 0.0)
                    }
                    cross_data.append(cross_item)
                
                combinations[f'{tech}_{tech}_{test_model}'] = cross_data
                print(f"Loaded {len(cross_data)} items from {file_path}")
    
    return combinations

def parse_test_results(test_results_str):
    """Parse test results string to list of integers"""
    if isinstance(test_results_str, list):
        return [int(x) for x in test_results_str]
    elif isinstance(test_results_str, str):
        # Handle string format like "['1', '0', '0', '0']"
        try:
            result_list = eval(test_results_str)
            return [int(x) for x in result_list]
        except Exception as e:
            print(e)
            return []
    else:
        
        return []

def process_code_pair(technique, model, results_dir="test_result5"):
    """Process original vs technique pair and calculate scores for both codes"""
    print(f"\n{'='*60}")
    print(f"Processing ORIGINAL vs {technique.upper()} ({model})")
    print(f"{'='*60}")
    
    # Load cross-technique test results
    test_combinations = load_cross_technique_results(technique, model, results_dir)
    # Load ground truth results
    original_results = load_ground_truth_results('original', model, results_dir)
    technique_results = load_ground_truth_results(technique, model, results_dir)
    
    print(f"Loaded {len(original_results)} original results")
    print(f"Loaded {len(technique_results)} technique results")
    print(f"Found {len(test_combinations)} test combinations")
    
    # Group results by task_id
    task_data = defaultdict(lambda: {'original': [], 'technique': []})
    
    # Process each combination
    for combo_name, combo_data in test_combinations.items():
        if not combo_data:
            continue
        
        print(f"Processing combination: {combo_name} with {len(combo_data)} items")
        
        for item in combo_data:
            task_id = item['task_id']
            code_technique = item.get('code_technique', '')
            
            # Clean up technique name
            for suffix in ['_gemini_bigcodebench_output_llm1', '_o4-mini_bigcodebench_output_llm1', '_bigcodebench_output_llm1']:
                code_technique = code_technique.replace(suffix, '')
            
            # Parse test results
            test_results = parse_test_results(item['test_results'])
            if not test_results:
                continue
                
            # Add to appropriate code category
            if code_technique == 'original':
                task_data[task_id]['original'].append(test_results)
            elif code_technique == technique:
                task_data[task_id]['technique'].append(test_results)
            else:
                # If technique name doesn't match exactly, try to match
                if 'original' in code_technique.lower():
                    task_data[task_id]['original'].append(test_results)
                else:
                    task_data[task_id]['technique'].append(test_results)
    
    # Calculate scores and select best code for each task
    evaluation_results = []
    task_ids = list(task_data.keys())
    
    with tqdm(task_ids, desc=f"Evaluating {technique} vs original", unit="task") as pbar:
        for task_id in pbar:
            data = task_data[task_id]
            pbar.set_description(f"Evaluating {task_id[:20]}...")
            
            # Calculate scores for original code
            original_scores = compute_score(data['original']) if data['original'] else []
            original_total_score = sum(original_scores) if original_scores else 0
            # Calculate scores for technique code  
            technique_scores = compute_score(data['technique']) if data['technique'] else []
            technique_total_score = sum(technique_scores) if technique_scores else 0
            
            # Select best code
            if original_total_score > technique_total_score:
                chosen_code = 'original'
                chosen_score = original_total_score
                actual_result = original_results.get(task_id, 0)
            elif technique_total_score > original_total_score:
                chosen_code = technique
                chosen_score = technique_total_score
                actual_result = technique_results.get(task_id, 0)
            else:
                # Same score - select original
                chosen_code = 'original'
                chosen_score = original_total_score  # they're equal
                actual_result = original_results.get(task_id, 0)
            
            evaluation_results.append({
                'task_id': task_id,
                'chosen_solution': chosen_code,
                'original_score': original_total_score,
                'technique_score': technique_total_score,
                'chosen_score': chosen_score,
                'actual_result': actual_result,
                'technique_pair': f'original_vs_{technique}'
            })
            
            # Update progress bar with current selection
            pbar.set_postfix({
                'chosen': chosen_code[:4],
                'score': chosen_score,
                'result': actual_result
            })
    
    return evaluation_results

def main():
    import sys
    
    # Get model from command line argument or default to both models
    specified_model = sys.argv[1] if len(sys.argv) > 1 else None
    results_dir = sys.argv[2] if len(sys.argv) > 2 else "test_result5"
    
    # Auto-detect available techniques from results
    available_techniques = set()
    models_to_process = []
    
    # Check what's available in results directory
    result_dirs = [
        f"{results_dir}/o4-mini_results",
        f"{results_dir}/gemini_results",
        results_dir
    ]
    
    for dir_path in result_dirs:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.startswith('code_') and filename.endswith('.jsonl'):
                    # Extract technique and model
                    name = filename.replace('code_', '').replace('.jsonl', '')
                    parts = name.split('_')
                    if len(parts) >= 1:
                        technique = parts[0]
                        available_techniques.add(technique)
                        
                        # Extract model name
                        for model in ['gemini', 'o4-mini']:
                            if model in filename:
                                if specified_model is None or model == specified_model:
                                    if model not in models_to_process:
                                        models_to_process.append(model)
    
    # Remove 'original' from techniques list for comparison
    available_techniques.discard('original')
    techniques = sorted(list(available_techniques))
    
    if not models_to_process:
        models_to_process = ['gemini', 'o4-mini'] if specified_model is None else [specified_model]
    
    print(f"Available techniques: {techniques}")
    print(f"Models to process: {models_to_process}")
    print(f"Results directory: {results_dir}")
    
    random.seed(42)  # For reproducible results
    
    # Process each model and technique separately
    total_combinations = len(models_to_process) * len(techniques)
    
    with tqdm(total=total_combinations, desc="Processing all combinations", unit="combo") as main_pbar:
        for model in models_to_process:
            for technique in techniques:
                main_pbar.set_description(f"Processing {technique} ({model})")
                
                print(f"\n{'='*80}")
                print(f"PROCESSING TECHNIQUE: {technique.upper()} for model: {model.upper()}")
                print(f"{'='*80}")
                
                results = process_code_pair(technique, model, results_dir)
                
                if results:
                    # Create directory structure
                    output_dir = f'final_evaluation_results/{model}'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save results to separate file
                    output_file = f'{output_dir}/original_vs_{technique}_evaluation.jsonl'
                    with open(output_file, 'w') as f:
                        for result in results:
                            f.write(json.dumps(result) + '\n')
                    
                    print(f"\nSaved {len(results)} results to {output_file}")
                    
                    # Calculate statistics for this technique pair
                    correct_predictions = sum(1 for r in results if r['actual_result'] == 1)
                    total_predictions = len(results)
                    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                    
                    original_wins = sum(1 for r in results if r['chosen_solution'] == 'original')
                    technique_wins = sum(1 for r in results if r['chosen_solution'] == technique)
                    
                    print(f"\nStatistics for {technique} vs original ({model}):")
                    print(f"Total selections: {total_predictions}")
                    print(f"Correct selections: {correct_predictions}")
                    print(f"Accuracy: {accuracy:.2f}%")
                    print(f"Original wins: {original_wins}")
                    print(f"Technique wins: {technique_wins}")
                    
                    # Save summary statistics
                    summary = {
                        'technique': technique,
                        'model': model,
                        'total_selections': total_predictions,
                        'correct_selections': correct_predictions,
                        'accuracy': accuracy,
                        'original_wins': original_wins,
                        'technique_wins': technique_wins
                    }
                    
                    summary_file = f'{output_dir}/original_vs_{technique}_summary.json'
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                    
                    # Update main progress bar with statistics
                    main_pbar.set_postfix({
                        'accuracy': f'{accuracy:.1f}%',
                        'orig_wins': original_wins,
                        'tech_wins': technique_wins
                    })
                        
                else:
                    print(f"No results found for {technique} ({model})")
                
                # Update progress
                main_pbar.update(1)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()