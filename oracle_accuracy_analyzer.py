import json
import os
from collections import defaultdict

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

def analyze_oracle_accuracy(technique, model):
    """Analyze what accuracy we'd get if we always picked the variant with test_result=1"""
    
    print(f"\n{'='*70}")
    print(f"Oracle Analysis: ORIGINAL vs {technique.upper()} ({model})")
    print(f"{'='*70}")
    
    # Load ground truth from test_result5
    ground_truth = load_ground_truth_results(model)
    
    # Get ground truth for original and technique separately
    original_ground_truth = {}
    technique_ground_truth = {}
    
    for task_id, result in ground_truth.items():
        # Assume original results are from files with 'original' in name
        # and technique results are from files with technique name
        # This is a simplification - we'd need to check file names more carefully
        original_ground_truth[task_id] = result
        technique_ground_truth[task_id] = result
    
    # Load ground truth more carefully by checking file names
    model_dir = f'test_result5/{model}_results'
    original_ground_truth = {}
    technique_ground_truth = {}
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(model_dir, filename)
                data = load_jsonl(file_path)
                
                # Check if this file is for original or technique
                is_original = 'original' in filename.lower()
                is_technique = technique in filename.lower()
                
                for item in data:
                    task_id = item.get('task_id') or item.get('id')
                    if task_id:
                        test_result = item.get('test_result', 0)
                        
                        if is_original:
                            original_ground_truth[task_id] = test_result
                        elif is_technique:
                            technique_ground_truth[task_id] = test_result
    
    print(f"Original ground truth entries: {len(original_ground_truth)}")
    print(f"Technique ground truth entries: {len(technique_ground_truth)}")
    
    # Find common tasks
    common_tasks = set(original_ground_truth.keys()) & set(technique_ground_truth.keys())
    print(f"Common tasks: {len(common_tasks)}")
    
    if not common_tasks:
        print("No common tasks found. Let's analyze all available ground truth data...")
        # Use all ground truth data and assume both variants have same results
        common_tasks = set(ground_truth.keys())
        original_ground_truth = ground_truth.copy()
        technique_ground_truth = ground_truth.copy()
    
    # Oracle selection: always pick the variant with test_result=1 if available
    oracle_results = []
    selection_stats = {'original_selected': 0, 'technique_selected': 0, 'both_pass': 0, 'both_fail': 0, 'tie_random': 0}
    
    for task_id in common_tasks:
        original_result = original_ground_truth.get(task_id, 0)
        technique_result = technique_ground_truth.get(task_id, 0)
        
        # Oracle selection logic
        if original_result == 1 and technique_result == 0:
            # Only original passes
            chosen = 'original'
            chosen_result = original_result
            selection_stats['original_selected'] += 1
        elif technique_result == 1 and original_result == 0:
            # Only technique passes  
            chosen = technique
            chosen_result = technique_result
            selection_stats['technique_selected'] += 1
        elif original_result == 1 and technique_result == 1:
            # Both pass - pick original (arbitrary choice)
            chosen = 'original'
            chosen_result = original_result
            selection_stats['both_pass'] += 1
        elif original_result == 0 and technique_result == 0:
            # Both fail - pick original (arbitrary choice)
            chosen = 'original' 
            chosen_result = original_result
            selection_stats['both_fail'] += 1
        else:
            # Shouldn't reach here, but handle anyway
            chosen = 'original'
            chosen_result = original_result
            selection_stats['tie_random'] += 1
        
        oracle_results.append({
            'task_id': task_id,
            'original_result': original_result,
            'technique_result': technique_result,
            'chosen': chosen,
            'chosen_result': chosen_result
        })
    
    # Calculate oracle accuracy
    correct_selections = sum(1 for r in oracle_results if r['chosen_result'] == 1)
    total_tasks = len(oracle_results)
    oracle_accuracy = (correct_selections / total_tasks * 100) if total_tasks > 0 else 0
    
    print(f"\n=== ORACLE ANALYSIS RESULTS ===")
    print(f"Total tasks analyzed: {total_tasks}")
    print(f"Oracle accuracy: {oracle_accuracy:.1f}% ({correct_selections}/{total_tasks})")
    print(f"")
    print(f"Selection breakdown:")
    print(f"  Original selected (only original passes): {selection_stats['original_selected']}")
    print(f"  Technique selected (only technique passes): {selection_stats['technique_selected']}")
    print(f"  Both pass (chose original): {selection_stats['both_pass']}")
    print(f"  Both fail (chose original): {selection_stats['both_fail']}")
    print(f"  Ties/other (chose original): {selection_stats['tie_random']}")
    
    # Show some examples
    print(f"\n=== EXAMPLES ===")
    examples = oracle_results[:10]  # First 10
    for ex in examples:
        print(f"Task {ex['task_id']}: Original={ex['original_result']}, {technique}={ex['technique_result']} → Chose {ex['chosen']} (result={ex['chosen_result']})")
    
    return oracle_accuracy, oracle_results, selection_stats

def main():
    model = 'gemini'
    
    # Test techniques
    techniques = ['active_tense', 'declarative_to_interrogative', 'adversarial_function_name']
    
    print(f"=== ORACLE ACCURACY ANALYSIS FOR {model.upper()} ===")
    print("This analysis shows what accuracy we'd get if we always")
    print("selected the code variant with test_result=1 when available.\n")
    
    all_results = {}
    
    for technique in techniques:
        try:
            accuracy, results, stats = analyze_oracle_accuracy(technique, model)
            all_results[technique] = {
                'accuracy': accuracy,
                'total_tasks': len(results),
                'stats': stats
            }
        except Exception as e:
            print(f"Error analyzing {technique}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ORACLE ACCURACIES")
    print(f"{'='*70}")
    
    for technique, data in all_results.items():
        print(f"{technique:25}: {data['accuracy']:6.1f}% ({data['total_tasks']} tasks)")
    
    if all_results:
        avg_accuracy = sum(data['accuracy'] for data in all_results.values()) / len(all_results)
        print(f"{'Average':25}: {avg_accuracy:6.1f}%")

if __name__ == "__main__":
    main()