import json
import os
from collections import defaultdict

def debug_data_loading():
    """Debug function to check data loading"""
    model = 'gemini'
    
    print("=== DEBUGGING DATA LOADING ===")
    
    # Check ground truth loading
    print(f"\n1. Checking ground truth data for {model}...")
    ground_truth_data = {}
    model_dir = f'test_result5/{model}_results'
    
    if os.path.exists(model_dir):
        print(f"Found directory: {model_dir}")
        files = [f for f in os.listdir(model_dir) if f.endswith('.jsonl')]
        print(f"JSONL files found: {files}")
        
        for filename in files[:2]:  # Just check first 2 files
            file_path = os.path.join(model_dir, filename)
            print(f"\nLoading {filename}...")
            try:
                with open(file_path, 'r') as f:
                    count = 0
                    for line in f:
                        item = json.loads(line)
                        task_id = item.get('task_id') or item.get('id')
                        if task_id and count < 3:  # Show first 3 entries
                            print(f"  Task: {task_id}, Result: {item.get('test_result', 'N/A')}")
                            ground_truth_data[task_id] = item.get('test_result', 0)
                            count += 1
                        if count >= 3:
                            break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    else:
        print(f"Directory not found: {model_dir}")
    
    print(f"\nGround truth loaded: {len(ground_truth_data)} entries")
    print(f"Sample entries: {dict(list(ground_truth_data.items())[:3])}")
    
    # Check cross-technique data
    print(f"\n2. Checking cross-technique data for {model}...")
    cross_dir = f'{model}_cross_results'
    
    if os.path.exists(cross_dir):
        print(f"Found directory: {cross_dir}")
        files = [f for f in os.listdir(cross_dir) if f.endswith('.jsonl')]
        print(f"Cross-technique files found: {len(files)}")
        
        # Check first file
        if files:
            first_file = files[0]
            print(f"\nChecking {first_file}...")
            file_path = os.path.join(cross_dir, first_file)
            
            try:
                with open(file_path, 'r') as f:
                    count = 0
                    for line in f:
                        item = json.loads(line)
                        task_id = item.get('task_id')
                        if task_id and count < 3:
                            print(f"  Task: {task_id}")
                            print(f"  Code technique: {item.get('code_technique', 'N/A')}")
                            print(f"  Test technique: {item.get('test_technique', 'N/A')}")
                            print(f"  Test results: {item.get('test_results', 'N/A')}")
                            print(f"  Pass rate: {item.get('pass_rate', 'N/A')}")
                            print("  ---")
                            count += 1
                        if count >= 3:
                            break
            except Exception as e:
                print(f"Error loading {first_file}: {e}")
    else:
        print(f"Directory not found: {cross_dir}")
    
    # Check technique extraction
    print(f"\n3. Testing technique extraction...")
    filename = "code_active_tense_gemini_bigcodebench_output_llm1_v2_test_declarative_to_interrogative_gemini_bigcodebench_output_llm1_v2.jsonl"
    parts = filename.replace('.jsonl', '').split('_test_')
    if len(parts) == 2:
        code_part = parts[0].replace('code_', '')
        test_part = parts[1]
        
        code_technique = code_part.replace(f'_{model}_bigcodebench_output_llm1_v2', '')
        test_technique = test_part.replace(f'_{model}_bigcodebench_output_llm1_v2', '')
        
        print(f"Filename: {filename}")
        print(f"Code technique: {code_technique}")
        print(f"Test technique: {test_technique}")
        print(f"Key would be: {code_technique}_vs_{test_technique}")

if __name__ == "__main__":
    debug_data_loading()