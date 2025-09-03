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

def get_technique_file_mapping(model):
    """Get mapping from technique names to actual file names"""
    mapping = {
        'gemini': {
            'original': 'code_original_gemini_bigcodebench_output_llm1_v2.jsonl',
            'active_tense': 'code_active_gemini_gemini_bigcodebench_output_llm1_v2.jsonl',
            'adversarial_function_name': 'code_adversarial_gemini_name_gemini_bigcodebench_output_llm1_v2.jsonl',
            'declarative_to_interrogative': 'code_declarative_gemini_interrogative_gemini_bigcodebench_output_llm1_v2.jsonl',
            'future_tense': 'code_future_gemini_gemini_bigcodebench_output_llm1_v2.jsonl',
            'past_tense': 'code_past_gemini_gemini_bigcodebench_output_llm1_v2.jsonl',
            'passive_tense': 'code_passive_gemini_gemini_bigcodebench_output_llm1_v2.jsonl'
        }
    }
    
    return mapping.get(model, {})

def load_ground_truth_for_technique(technique, model):
    """Load ground truth results for a specific technique"""
    ground_truth_data = {}
    
    file_mapping = get_technique_file_mapping(model)
    filename = file_mapping.get(technique)
    
    if not filename:
        print(f"Warning: No file mapping found for technique '{technique}' in model '{model}'")
        return ground_truth_data
    
    file_path = f'test_result5/{model}_results/{filename}'
    
    if os.path.exists(file_path):
        data = load_jsonl(file_path)
        for item in data:
            task_id = item.get('task_id') or item.get('id')
            if task_id:
                test_result = item.get('test_result', 0)
                ground_truth_data[task_id] = test_result
    else:
        print(f"Warning: File not found: {file_path}")
    
    return ground_truth_data

def analyze_oracle_accuracy(technique, model):
    """Analyze what accuracy we'd get if we always picked the variant with test_result=1"""
    
    print(f"\n{'='*70}")
    print(f"Oracle Analysis: ORIGINAL vs {technique.upper()} ({model})")
    print(f"{'='*70}")
    
    # Load ground truth for both original and technique
    original_ground_truth = load_ground_truth_for_technique('original', model)
    technique_ground_truth = load_ground_truth_for_technique(technique, model)
    
    print(f"Original ground truth entries: {len(original_ground_truth)}")
    print(f"Technique ground truth entries: {len(technique_ground_truth)}")
    
    # Find common tasks
    common_tasks = set(original_ground_truth.keys()) & set(technique_ground_truth.keys())
    print(f"Common tasks: {len(common_tasks)}")
    
    if not common_tasks:
        print("No common tasks found!")
        return 0, [], {}
    
    # Oracle selection: always pick the variant with test_result=1 if available
    oracle_results = []
    selection_stats = {
        'original_selected': 0,
        'technique_selected': 0, 
        'both_pass': 0,
        'both_fail': 0,
        'different_results': 0
    }
    
    for task_id in common_tasks:
        original_result = original_ground_truth[task_id]
        technique_result = technique_ground_truth[task_id]
        
        # Oracle selection logic
        if original_result == 1 and technique_result == 0:
            # Only original passes
            chosen = 'original'
            chosen_result = 1
            selection_stats['original_selected'] += 1
            selection_stats['different_results'] += 1
        elif technique_result == 1 and original_result == 0:
            # Only technique passes  
            chosen = technique
            chosen_result = 1
            selection_stats['technique_selected'] += 1
            selection_stats['different_results'] += 1
        elif original_result == 1 and technique_result == 1:
            # Both pass - pick original (arbitrary choice, both would give correct result)
            chosen = 'original'
            chosen_result = 1
            selection_stats['both_pass'] += 1
        else:  # both are 0
            # Both fail - pick original (arbitrary choice, both would give wrong result)
            chosen = 'original' 
            chosen_result = 0
            selection_stats['both_fail'] += 1
        
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
    print(f"  Tasks where results differ: {selection_stats['different_results']}")
    
    # Calculate potential improvement
    original_baseline = sum(1 for r in oracle_results if r['original_result'] == 1)
    technique_baseline = sum(1 for r in oracle_results if r['technique_result'] == 1)
    
    print(f"")
    print(f"Baseline accuracies:")
    print(f"  Always choose original: {original_baseline/total_tasks*100:.1f}% ({original_baseline}/{total_tasks})")
    print(f"  Always choose technique: {technique_baseline/total_tasks*100:.1f}% ({technique_baseline}/{total_tasks})")
    print(f"  Oracle (perfect selection): {oracle_accuracy:.1f}% ({correct_selections}/{total_tasks})")
    
    # Show some examples where results differ
    different_examples = [r for r in oracle_results if r['original_result'] != r['technique_result']]
    if different_examples:
        print(f"\n=== EXAMPLES WHERE RESULTS DIFFER ===")
        for ex in different_examples[:5]:  # First 5 different cases
            print(f"Task {ex['task_id']}: Original={ex['original_result']}, {technique}={ex['technique_result']} → Oracle chose {ex['chosen']}")
    
    return oracle_accuracy, oracle_results, selection_stats

def main():
    model = 'gemini'
    
    # Available techniques based on file mapping
    file_mapping = get_technique_file_mapping(model)
    techniques = [t for t in file_mapping.keys() if t != 'original']
    
    print(f"=== ORACLE ACCURACY ANALYSIS FOR {model.upper()} ===")
    print("This analysis shows what accuracy we'd get if we always")
    print("selected the code variant with test_result=1 when available.\n")
    
    all_results = {}
    
    for technique in techniques:
        try:
            accuracy, results, stats = analyze_oracle_accuracy(technique, model)
            if results:  # Only store if we got results
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
        orig_wins = data['stats']['original_selected'] + data['stats']['both_pass'] + data['stats']['both_fail']
        tech_wins = data['stats']['technique_selected']
        different = data['stats']['different_results']
        
        print(f"{technique:25}: {data['accuracy']:6.1f}% | Diff results: {different:3d} | Technique wins: {tech_wins:3d}")
    
    if all_results:
        avg_accuracy = sum(data['accuracy'] for data in all_results.values()) / len(all_results)
        print(f"{'Average':25}: {avg_accuracy:6.1f}%")

if __name__ == "__main__":
    main()