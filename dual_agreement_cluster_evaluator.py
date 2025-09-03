import json
import random
import math
import os
import sys
from collections import Counter, defaultdict
from code_quality_analyzer import compare_code_quality_detailed

def compute_dual_agreement_score(execution_results_a, execution_results_b):
    """
    Compute dual agreement score using clustering approach with two code variants:
    Groups execution results by test case patterns and calculates scores based on
    test coverage and solution agreement within clusters for both code variants.
    Returns scores for both variants using: case_set_score × sqrt(solution_set_size)
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

def extract_technique_pairs(cross_results):
    """Extract unique technique pairs from cross results"""
    techniques = set()
    pairs = set()
    
    for key in cross_results.keys():
        code_tech, test_tech = key.split('_vs_')
        techniques.add(code_tech)
        techniques.add(test_tech)
        pairs.add((code_tech, test_tech))
    
    return list(techniques), list(pairs)

def process_dual_agreement(technique, model):
    """Process dual agreement between original vs technique using cluster-based scoring"""
    print(f"\n{'='*70}")
    print(f"Processing DUAL AGREEMENT: ORIGINAL vs {technique.upper()} ({model})")
    print(f"{'='*70}")
    
    # Load cross technique data
    cross_results = load_cross_technique_data(model)
    
    # Load ground truth
    ground_truth = load_ground_truth_results(model)
    
    # Group results by task_id for the 4 combinations
    task_data = defaultdict(lambda: {
        'original_original': [],    # original code + original test  
        'original_technique': [],   # original code + technique test
        'technique_original': [],   # technique code + original test
        'technique_technique': []   # technique code + technique test
    })
    
    # Collect the 4 specific cross-technique combinations
    target_combinations = [
        ('original', 'original'),
        ('original', technique),
        (technique, 'original'), 
        (technique, technique)
    ]
    
    for code_tech, test_tech in target_combinations:
        key = f"{code_tech}_vs_{test_tech}"
        if key in cross_results:
            data = cross_results[key]
            # Map to standardized combination keys
            if code_tech == 'original' and test_tech == 'original':
                combination_key = 'original_original'
            elif code_tech == 'original' and test_tech == technique:
                combination_key = 'original_technique'
            elif code_tech == technique and test_tech == 'original':
                combination_key = 'technique_original'
            elif code_tech == technique and test_tech == technique:
                combination_key = 'technique_technique'
            else:
                continue  # Skip if not one of our 4 target combinations
            
            for item in data:
                task_id = item.get('task_id')
                if task_id:
                    task_data[task_id][combination_key].append(item)
    
    # Calculate dual agreement scores for each task
    evaluation_results = []
    
    for task_id, data in task_data.items():
        # Check if we have all 4 combinations
        if not all(data[combo] for combo in ['original_original', 'original_technique', 
                                           'technique_original', 'technique_technique']):
            continue
            
        print(f"\nProcessing task_id {task_id}:")
        print(f"  Found all 4 combinations for {technique}")
        
        # Group results by code type for dual agreement scoring
        original_results = data['original_original'] + data['original_technique']
        technique_results = data['technique_original'] + data['technique_technique']
        
        # Compute dual agreement scores
        agreement_results = compute_dual_agreement_score(
            original_results, 
            technique_results
        )
        
        original_score = agreement_results['variant_a_score']
        technique_score = agreement_results['variant_b_score']
        
        print(f"  Original cluster score: {original_score:.3f}")
        print(f"  {technique} cluster score: {technique_score:.3f}")
        
        # Select best technique based on dual agreement scores
        if original_score > technique_score:
            chosen_technique = 'original'
            chosen_score = original_score
            selection_method = 'dual_agreement'
            quality_analysis = None
        elif technique_score > original_score:
            chosen_technique = technique
            chosen_score = technique_score
            selection_method = 'dual_agreement'
            quality_analysis = None
        else:
            # Tie - use code quality analysis for tie-breaking
            print(f"  Dual agreement tie detected. Using code quality analysis for task {task_id}")
            quality_analysis = compare_code_quality_detailed(task_id, 'original', technique, model)
            
            if quality_analysis['winner'] == 'original':
                chosen_technique = 'original'
                selection_method = 'quality_based'
            elif quality_analysis['winner'] == technique:
                chosen_technique = technique
                selection_method = 'quality_based'
            else:
                # Quality tie or error - fall back to random selection
                chosen_technique = random.choice(['original', technique])
                selection_method = 'random'
            
            chosen_score = original_score  # they're equal
            print(f"  Quality analysis result: {quality_analysis['winner']}, selected: {chosen_technique}")
        
        # Get actual result from ground truth
        actual_result = ground_truth.get(task_id, 0)
        
        result_entry = {
            'task_id': task_id,
            'original_technique': 'original',
            'variant_technique': technique,
            'original_score': original_score,
            'technique_score': technique_score,
            'chosen_technique': chosen_technique,
            'chosen_score': chosen_score,
            'actual_result': actual_result,
            'selection_method': selection_method,
            'combination_details': {
                'original_original_count': len(data['original_original']),
                'original_technique_count': len(data['original_technique']),
                'technique_original_count': len(data['technique_original']),
                'technique_technique_count': len(data['technique_technique'])
            },
            'agreement_details': agreement_results
        }
        
        # Add quality analysis details if it was used
        if quality_analysis is not None:
            result_entry['quality_analysis'] = quality_analysis
            
        evaluation_results.append(result_entry)
        
        print(f"  Selected: {chosen_technique}, Score: {chosen_score:.3f}, Actual result: {actual_result}")
    
    return evaluation_results

def main():
    # Get model from command line argument or default to gemini
    model = sys.argv[1] if len(sys.argv) > 1 else 'gemini'
    # Optional: limit number of technique pairs for testing
    max_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Available techniques (based on your data structure)
    techniques = [
        'active_tense', 
        'adversarial_function_name',
        'declarative_to_interrogative',
        'future_tense',
        'highlight_cap',
        'highlight_star',
        'lowercase_to_uppercase',
        'original',
        'passive_tense',
        'past_tense',
        'random_function_name',
        'remove_function_name',
        'rename_variables',
        'rephrase_prompt',
        'task_function_name',
        'unrelated_function_name',
        'uppercase_to_lowercase',
        'verb_to_similar_verb'
    ]
    
    random.seed(42)  # For reproducible results
    
    # Create output directory
    output_dir = f'dual_agreement_results/{model}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each technique against original
    all_results = []
    processed_count = 0
    
    for technique in techniques:
        if technique == 'original':  # Skip original vs original
            continue
            
        if max_pairs and processed_count >= max_pairs:
            break
            
        try:
            results = process_dual_agreement(technique, model)
            
            if results:
                # Save results for this technique
                output_file = f'{output_dir}/original_vs_{technique}_dual_agreement.jsonl'
                with open(output_file, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
                
                print(f"\nSaved {len(results)} results to {output_file}")
                
                # Calculate statistics
                correct_predictions = sum(1 for r in results if r['actual_result'] == 1)
                total_predictions = len(results)
                accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                
                original_wins = sum(1 for r in results if r['chosen_technique'] == 'original')
                technique_wins = sum(1 for r in results if r['chosen_technique'] == technique)
                
                print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
                print(f"Original wins: {original_wins}, {technique} wins: {technique_wins}")
                
                all_results.extend(results)
                processed_count += 1
            else:
                print(f"No results found for original vs {technique}")
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing original vs {technique}: {e}")
            processed_count += 1
            continue
    
    # Save comprehensive results
    if all_results:
        comprehensive_file = f'{output_dir}/comprehensive_dual_agreement_results.jsonl'
        with open(comprehensive_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')
        print(f"\nSaved comprehensive results with {len(all_results)} entries to {comprehensive_file}")
        
        # Calculate overall statistics
        total_correct = sum(1 for r in all_results if r['actual_result'] == 1)
        overall_accuracy = (total_correct / len(all_results) * 100) if all_results else 0
        print(f"Overall accuracy: {overall_accuracy:.1f}% ({total_correct}/{len(all_results)})")

if __name__ == "__main__":
    main()