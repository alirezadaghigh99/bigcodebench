import json
import os
from collections import defaultdict

def test_4_combinations(technique, model='gemini'):
    """Test that we can find the 4 cross-technique combinations"""
    
    print(f"=== Testing 4 combinations for ORIGINAL vs {technique} ({model}) ===\n")
    
    # Load cross technique data
    cross_results = {}
    cross_dir = f'{model}_cross_results'
    
    if not os.path.exists(cross_dir):
        print(f"Error: Directory not found: {cross_dir}")
        return
    
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
                
                key = f"{code_technique}_vs_{test_technique}"
                cross_results[key] = filename
    
    # Look for the 4 specific combinations
    target_combinations = [
        ('original', 'original'),
        ('original', technique),
        (technique, 'original'), 
        (technique, technique)
    ]
    
    found_files = {}
    missing_combinations = []
    
    for code_tech, test_tech in target_combinations:
        key = f"{code_tech}_vs_{test_tech}"
        combination_name = f"{code_tech}_{test_tech}"
        
        if key in cross_results:
            found_files[combination_name] = cross_results[key]
            print(f"✓ Found {combination_name}: {cross_results[key]}")
            
            # Load and show sample data
            file_path = os.path.join(cross_dir, cross_results[key])
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    if first_line:
                        item = json.loads(first_line)
                        print(f"  Sample task: {item.get('task_id', 'N/A')}")
                        print(f"  Test results: {item.get('test_results', 'N/A')[:5]}...")  # Show first 5
                        print(f"  Pass rate: {item.get('pass_rate', 'N/A')}")
            except Exception as e:
                print(f"  Error reading file: {e}")
        else:
            missing_combinations.append(combination_name)
            print(f"✗ Missing {combination_name}: {key}")
        
        print()
    
    # Summary
    print(f"=== SUMMARY ===")
    print(f"Found: {len(found_files)}/4 combinations")
    print(f"Missing: {missing_combinations}")
    
    if len(found_files) == 4:
        print("✅ All 4 combinations found! Ready for dual agreement analysis.")
        
        # Test data loading for first task
        print(f"\n=== Testing data loading for first task ===")
        task_data = defaultdict(lambda: {
            'original_original': [],
            'original_technique': [],
            'technique_original': [],
            'technique_technique': []
        })
        
        for combo_name, filename in found_files.items():
            file_path = os.path.join(cross_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline()
                    if first_line:
                        item = json.loads(first_line)
                        task_id = item.get('task_id')
                        if task_id:
                            task_data[task_id][combo_name].append(item)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        # Show results for first task
        if task_data:
            first_task = list(task_data.keys())[0]
            print(f"Task {first_task}:")
            for combo_name in ['original_original', 'original_technique', 'technique_original', 'technique_technique']:
                count = len(task_data[first_task][combo_name])
                print(f"  {combo_name}: {count} results")
    else:
        print("❌ Missing combinations. Check cross-technique data generation.")

if __name__ == "__main__":
    test_4_combinations('active_tense')
    print("\n" + "="*60 + "\n")
    test_4_combinations('declarative_to_interrogative')