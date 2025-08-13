#!/usr/bin/env python3

import json
import os
import tempfile
import re
import ast
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import traceback
from tqdm import tqdm

def extract_function_name(code: str) -> Optional[str]:
    """Extract the main function name from Python code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        # Fallback regex method
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
    return None

def convert_function_to_test_func(code: str) -> str:
    """Convert the main function name to test_func."""
    original_name = extract_function_name(code)
    if not original_name:
        return code
    
    # Replace function definition
    pattern = rf'\bdef\s+{re.escape(original_name)}\s*\('
    replacement = 'def test_func('
    converted_code = re.sub(pattern, replacement, code)
    
    return converted_code

def safe_execute_code(code: str, test_code: str, task_id: str) -> Dict[str, Any]:
    """Safely execute code with test in a temporary file using test_environment."""
    result = {
        'task_id': task_id,
        'test_result': 0,
        'failure_reason': None
    }
    
    # Path to test environment python
    test_env_python = "/Users/aliredaq/Downloads/bigcodebench/test_environment/bin/python"
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file.write('\n\n')
            temp_file.write(test_code)
            temp_file_path = temp_file.name
        
        try:
            # Run the code in a subprocess with timeout using test environment
            process = subprocess.run(
                [test_env_python, temp_file_path],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if process.returncode == 0:
                result['test_result'] = 1
                result['failure_reason'] = None
            else:
                result['test_result'] = 0
                result['failure_reason'] = f"Runtime error: {process.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            result['test_result'] = 0
            result['failure_reason'] = "Timeout: Code execution exceeded 60 seconds"
            
        except Exception as e:
            result['test_result'] = 0
            result['failure_reason'] = f"Execution error: {str(e)}"
            
    except Exception as e:
        result['test_result'] = 0
        result['failure_reason'] = f"Setup error: {str(e)}"
        
    finally:
        # Clean up temp file
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass
    
    return result

def load_bigcodebench_dataset() -> Dict[str, Any]:
    """Load the BigCodeBench dataset."""
    dataset_path = Path("/Users/aliredaq/Downloads/bigcodebench/dataset/bigcodebench.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    problems = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                problem = json.loads(line)
                problems[problem['task_id']] = problem
    
    return problems

def process_generation_file(file_path: str, problems: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a generation file and test each solution."""
    results = []
    
    # First pass to count lines
    with open(file_path, 'r') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    with open(file_path, 'r') as f:
        with tqdm(total=total_lines, desc=f"Processing {Path(file_path).name}") as pbar:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    sample = json.loads(line)
                    task_id = sample['task_id']
                    
                    if task_id not in problems:
                        pbar.set_postfix_str(f"Warning: Task {task_id} not found")
                        pbar.update(1)
                        continue
                    
                    # Get the generated code
                    response_code = sample.get('response_code', '')
                    if not response_code:
                        results.append({
                            'task_id': task_id,
                            'test_result': 0,
                            'failure_reason': 'No response_code found'
                        })
                        pbar.set_postfix_str(f"{task_id}: No code")
                        pbar.update(1)
                        continue
                    
                    # Convert function name to test_func
                    converted_code = convert_function_to_test_func(response_code)
                    
                    # Get test code from dataset
                    problem = problems[task_id]
                    test_code = problem['test']
                    
                    # Execute and test
                    result = safe_execute_code(converted_code, test_code, task_id)
                    results.append(result)
                    
                    status = 'PASS' if result['test_result'] == 1 else 'FAIL'
                    pbar.set_postfix_str(f"{task_id}: {status}")
                    pbar.update(1)
                    
                except Exception as e:
                    pbar.set_postfix_str(f"Error: {str(e)[:30]}")
                    pbar.update(1)
                    continue
    
    return results

def main():
    # Load dataset
    print("Loading BigCodeBench dataset...")
    problems = load_bigcodebench_dataset()
    print(f"Loaded {len(problems)} problems")
    
    # Process each model's generation files
    generation_root = Path("/Users/aliredaq/Downloads/bigcodebench/generation_output")
    
    for model_folder in generation_root.iterdir():
        if not model_folder.is_dir():
            continue
            
        print(f"\nProcessing model: {model_folder.name}")
        
        # Create test_generation folder
        test_gen_folder = model_folder / "test_generation"
        test_gen_folder.mkdir(exist_ok=True)
        
        # Process each code generation file
        for gen_file in model_folder.glob("code_*.jsonl"):
            print(f"  Processing {gen_file.name}")
            
            try:
                results = process_generation_file(str(gen_file), problems)
                
                # Save results
                output_file = test_gen_folder / f"test_{gen_file.name}"
                with open(output_file, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
                
                # Print summary
                passed = sum(1 for r in results if r['test_result'] == 1)
                total = len(results)
                print(f"    Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
                
            except Exception as e:
                print(f"    Error processing {gen_file.name}: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    main()