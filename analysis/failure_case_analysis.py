#!/usr/bin/env python3
"""
Script to analyze failure cases where dual agreement system failed (actual_result=0) 
but original approach succeeded (test_result=1). Select 15 instances per model.
"""

import json
import csv
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    data = []
    if not os.path.exists(file_path):
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def find_failure_instances(models, base_path):
    """Find instances where actual_result=0 but original test_result=1."""
    failure_instances = {}
    
    for model in models:
        print(f"\nProcessing {model}...")
        failure_instances[model] = []
        
        # Get original test results
        original_test_file = Path(base_path) / f"generation_output/{model}/test_generation/test_code_original_{model}_bigcodebench_output_llm1.jsonl"
        original_test_data = load_jsonl(original_test_file)
        original_test_dict = {item['task_id']: item for item in original_test_data}
        
        # Check all evaluation files for this model
        model_path = Path(base_path) / f"final_evaluation_results1/{model}"
        
        for eval_file in model_path.glob("original_vs_*_evaluation.jsonl"):
            technique = eval_file.stem.replace("original_vs_", "").replace("_evaluation", "")
            
            data = load_jsonl(eval_file)
            for item in data:
                task_id = item['task_id']
                original_test_item = original_test_dict.get(task_id, {})
                original_test_result = original_test_item.get('test_result', 0)  # Default to 0 if not found
                
                original_score = item.get('original_score', 0)
                chosen_score = item.get('chosen_score', 0)
                
                # Check if system failed (actual_result=0) but original succeeded (test_result=1)
                if (item.get('actual_result') == 0 and 
                    original_test_result == 1):
                    
                    failure_instances[model].append({
                        'task_id': task_id,
                        'technique': technique,
                        'chosen_solution': item.get('chosen_solution', ''),
                        'original_score': original_score,
                        'technique_score': item.get('technique_score', 0),
                        'chosen_score': chosen_score,
                        'score_difference': chosen_score - original_score,
                        'actual_result': item['actual_result'],
                        'original_test_result': original_test_result
                    })
    
    return failure_instances

def get_generation_data(task_id, model, technique, base_path):
    """Extract generation data for a specific task."""
    gen_path = Path(base_path) / f"generation_output/{model}"
    
    # Get original prompt and response
    original_file = gen_path / f"code_original_{model}_bigcodebench_output_llm1.jsonl"
    original_data = load_jsonl(original_file)
    original_item = next((item for item in original_data if item['task_id'] == task_id), None)
    
    # Get technique prompt and response
    technique_file = gen_path / f"code_{technique}_{model}_bigcodebench_output_llm1.jsonl"
    technique_data = load_jsonl(technique_file)
    technique_item = next((item for item in technique_data if item['task_id'] == task_id), None)
    
    # Get original test generation (both test result and generated test code)
    original_test_file = gen_path / "test_generation" / f"test_code_original_{model}_bigcodebench_output_llm1.jsonl"
    original_test_data = load_jsonl(original_test_file)
    original_test_item = next((item for item in original_test_data if item['task_id'] == task_id), None)
    
    # Get technique test generation (both test result and generated test code)
    technique_test_file = gen_path / "test_generation" / f"test_code_{technique}_{model}_bigcodebench_output_llm1.jsonl"
    technique_test_data = load_jsonl(technique_test_file)
    technique_test_item = next((item for item in technique_test_data if item['task_id'] == task_id), None)
    
    # Get original test prompt (from test generation files)
    original_test_prompt_file = gen_path / f"test_original_{model}_bigcodebench_output_llm1.jsonl"
    original_test_prompt_data = load_jsonl(original_test_prompt_file)
    original_test_prompt_item = next((item for item in original_test_prompt_data if item['task_id'] == task_id), None)
    
    # Get technique test prompt
    technique_test_prompt_file = gen_path / f"test_{technique}_{model}_bigcodebench_output_llm1.jsonl"
    technique_test_prompt_data = load_jsonl(technique_test_prompt_file)
    technique_test_prompt_item = next((item for item in technique_test_prompt_data if item['task_id'] == task_id), None)
    
    # Get cross test results (original code tested on original tests)
    cross_result_file = gen_path / "cross_result" / f"code_original_{model}_bigcodebench_output_llm1_test_original_{model}_bigcodebench_output_llm1.jsonl"
    cross_result_data = load_jsonl(cross_result_file)
    cross_result_item = next((item for item in cross_result_data if item['task_id'] == task_id), None)
    
    return {
        'original_item': original_item,
        'technique_item': technique_item,
        'original_test_item': original_test_item,
        'technique_test_item': technique_test_item,
        'original_test_prompt_item': original_test_prompt_item,
        'technique_test_prompt_item': technique_test_prompt_item,
        'cross_result_item': cross_result_item
    }

def extract_user_prompt(prompt_list):
    """Extract user prompt from prompt list."""
    if not prompt_list:
        return ""
    
    for item in prompt_list:
        if item.get('role') == 'user':
            return item.get('content', '')
    return ""

def extract_generated_test_code(response_text):
    """Extract generated test code from response."""
    if not response_text:
        return ""
    
    # Look for code blocks in the response
    import re
    
    # Try to find Python code blocks
    code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    # Try to find general code blocks
    code_blocks = re.findall(r'```\n(.*?)\n```', response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    
    return response_text

def main():
    base_path = "/Users/aliredaq/Downloads/bigcodebench"
    models = ['gemini', 'deepseek', 'o4-mini']
    
    # Find failure instances
    print("Finding failure instances where actual_result=0 but original test_result=1...")
    failure_instances = find_failure_instances(models, base_path)
    
    # Print summary
    total_found = 0
    for model, instances in failure_instances.items():
        print(f"{model}: {len(instances)} failure instances")
        total_found += len(instances)
    
    print(f"\nTotal instances found: {total_found}")
    
    # Select up to 15 instances per model
    selected_instances = {}
    for model, instances in failure_instances.items():
        # Sort by score difference (to understand the impact of the failure)
        instances.sort(key=lambda x: x['score_difference'])
        selected_instances[model] = instances[:15]
        print(f"Selected {len(selected_instances[model])} instances for {model}")
    
    # Create CSV data
    csv_data = []
    
    for model, instances in selected_instances.items():
        print(f"\nProcessing {model}...")
        
        for i, instance in enumerate(instances):
            task_id = instance['task_id']
            technique = instance['technique']
            
            print(f"  Processing {task_id} with technique {technique} (score difference: {instance['score_difference']})")
            
            # Get generation data
            data = get_generation_data(task_id, model, technique, base_path)
            
            # Extract information
            original_prompt = ""
            technique_prompt = ""
            original_response_code = ""
            technique_response_code = ""
            original_test_result = ""
            technique_test_result = ""
            original_test_prompt = ""
            technique_test_prompt = ""
            original_generated_test = ""
            technique_generated_test = ""
            cross_test_output = ""
            
            if data['original_item']:
                original_prompt = extract_user_prompt(data['original_item'].get('prompt', []))
                original_response_code = data['original_item'].get('response_code', '')
            
            if data['technique_item']:
                technique_prompt = extract_user_prompt(data['technique_item'].get('prompt', []))
                technique_response_code = data['technique_item'].get('response_code', '')
            
            if data['original_test_item']:
                original_test_result = data['original_test_item'].get('test_result', '')
            
            if data['technique_test_item']:
                technique_test_result = data['technique_test_item'].get('test_result', '')
            
            # Extract test prompts and generated test code
            if data['original_test_prompt_item']:
                original_test_prompt = extract_user_prompt(data['original_test_prompt_item'].get('prompt', []))
                original_generated_test = extract_generated_test_code(data['original_test_prompt_item'].get('response', ''))
            
            if data['technique_test_prompt_item']:
                technique_test_prompt = extract_user_prompt(data['technique_test_prompt_item'].get('prompt', []))
                technique_generated_test = extract_generated_test_code(data['technique_test_prompt_item'].get('response', ''))
            
            if data['cross_result_item']:
                cross_test_output = str(data['cross_result_item'].get('test_results', ''))
            
            # Add to CSV data
            csv_data.append({
                'model': model,
                'task_id': task_id,
                'technique': technique,
                'chosen_solution': instance['chosen_solution'],
                'original_score': instance['original_score'],
                'technique_score': instance['technique_score'],
                'chosen_score': instance['chosen_score'],
                'score_difference': instance['score_difference'],
                'actual_result': instance['actual_result'],
                'original_test_result': instance['original_test_result'],
                'original_prompt': original_prompt,
                'technique_prompt': technique_prompt,
                'original_response_code': original_response_code,
                'technique_response_code': technique_response_code,
                'technique_test_result': technique_test_result,
                'original_test_prompt': original_test_prompt,
                'technique_test_prompt': technique_test_prompt,
                'original_generated_test': original_generated_test,
                'technique_generated_test': technique_generated_test,
                'cross_test_output': cross_test_output
            })
    
    # Write to CSV
    output_file = Path(base_path) / "analysis" / "failure_case_analysis.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'model', 'task_id', 'technique', 'chosen_solution', 'original_score', 
            'technique_score', 'chosen_score', 'score_difference', 'actual_result',
            'original_test_result', 'original_prompt', 'technique_prompt', 
            'original_response_code', 'technique_response_code', 'technique_test_result',
            'original_test_prompt', 'technique_test_prompt', 'original_generated_test',
            'technique_generated_test', 'cross_test_output'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nCSV file saved to: {output_file}")
    print(f"Total instances: {len(csv_data)}")
    
    # Print score difference summary
    if csv_data:
        differences = [row['score_difference'] for row in csv_data]
        print(f"\nScore difference summary (chosen - original):")
        print(f"  Min difference: {min(differences)}")
        print(f"  Max difference: {max(differences)}")
        print(f"  Average difference: {sum(differences) / len(differences):.2f}")
        
        # Count negative vs positive differences
        negative_count = sum(1 for d in differences if d < 0)
        positive_count = sum(1 for d in differences if d > 0)
        zero_count = sum(1 for d in differences if d == 0)
        
        print(f"\nScore difference distribution:")
        print(f"  Negative (chosen < original): {negative_count}")
        print(f"  Zero (chosen = original): {zero_count}")
        print(f"  Positive (chosen > original): {positive_count}")
        
    print(f"\nColumns in the CSV:")
    for i, col in enumerate(fieldnames, 1):
        print(f"  {i:2d}. {col}")

if __name__ == "__main__":
    main()