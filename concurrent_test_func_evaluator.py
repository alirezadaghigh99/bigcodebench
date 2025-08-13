#!/usr/bin/env python3

import json
import os
import tempfile
import re
import ast
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import traceback
from tqdm import tqdm
import concurrent.futures
import threading
from functools import partial

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

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from response, handling various formats."""
    if not response:
        return None
    
    # Try to extract code from markdown blocks first
    code_block_pattern = r'```python\n(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Take the last match
    
    # If no markdown blocks, try to extract from plain text
    # Look for import statements as start indicators
    lines = response.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Start collecting when we see import or def
        if re.match(r'^\s*(import|from|def)', line):
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip() if code_lines else None

def fix_truncated_code(code: str) -> str:
    """Try to fix common truncation issues."""
    if not code:
        return code
    
    # Fix unterminated triple quotes
    triple_quote_count = code.count("'''")
    if triple_quote_count % 2 == 1:  # Odd number means unterminated
        code += "\n    '''"
    
    double_triple_quote_count = code.count('"""')
    if double_triple_quote_count % 2 == 1:
        code += '\n    """'
    
    # Check if we have incomplete function body
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('def ') and line.endswith(':'):
            # Check if there's any actual function body after this
            has_body = False
            for j in range(i+1, len(lines)):
                if lines[j].strip() and not lines[j].startswith('    '):
                    break
                if lines[j].strip() and lines[j].startswith('    '):
                    has_body = True
                    break
            
            if not has_body:
                # Add a minimal body
                code += "\n    pass"
                break
    
    return code

def validate_and_fix_code(response_code: str, response: str) -> str:
    """Validate and fix code, falling back to response if needed."""
    
    # First try the response_code as is
    try:
        compile(response_code, '<string>', 'exec')
        return response_code
    except SyntaxError:
        pass
    
    # Try to fix truncation issues
    fixed_code = fix_truncated_code(response_code)
    try:
        compile(fixed_code, '<string>', 'exec')
        return fixed_code
    except SyntaxError:
        pass
    
    # Try to extract from the full response
    extracted_code = extract_code_from_response(response)
    if extracted_code:
        try:
            compile(extracted_code, '<string>', 'exec')
            return extracted_code
        except SyntaxError:
            # Try fixing the extracted code too
            fixed_extracted = fix_truncated_code(extracted_code)
            try:
                compile(fixed_extracted, '<string>', 'exec')
                return fixed_extracted
            except SyntaxError:
                pass
    
    # If all else fails, return the original (will likely fail tests but at least we track it)
    return response_code

def convert_function_to_task_func(code: str) -> str:
    """Convert the main function name to task_func to match test expectations."""
    original_name = extract_function_name(code)
    if not original_name:
        return code
    
    # Replace function definition
    pattern = rf'\bdef\s+{re.escape(original_name)}\s*\('
    replacement = 'def task_func('
    converted_code = re.sub(pattern, replacement, code)
    
    # Replace function calls
    call_pattern = rf'\b{re.escape(original_name)}\s*\('
    call_replacement = 'task_func('
    converted_code = re.sub(call_pattern, call_replacement, converted_code)
    
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
        # Create temporary file with the code, test, and unittest.main()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file.write('\n\n')
            temp_file.write(test_code)
            temp_file.write('\n\nif __name__ == "__main__":\n    unittest.main()\n')
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
                stderr_output = process.stderr.strip()
                stdout_output = process.stdout.strip()
                
                # Combine both outputs for better debugging
                if stderr_output and stdout_output:
                    result['failure_reason'] = f"STDERR: {stderr_output}\nSTDOUT: {stdout_output}"
                elif stderr_output:
                    result['failure_reason'] = f"STDERR: {stderr_output}"
                elif stdout_output:
                    result['failure_reason'] = f"STDOUT: {stdout_output}"
                else:
                    result['failure_reason'] = f"Non-zero exit code: {process.returncode}"
                
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

def process_single_sample(sample_data: tuple, problems: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    """Process a single sample and return result."""
    line, line_idx = sample_data
    
    try:
        sample = json.loads(line)
        task_id = sample['task_id']
        
        if task_id not in problems:
            result = {
                'task_id': task_id,
                'test_result': 0,
                'failure_reason': f'Task {task_id} not found in dataset'
            }
            if progress_callback:
                progress_callback(task_id, "NOT_FOUND")
            return result
        
        # Get the generated code
        response_code = sample.get('response_code', '')
        response = sample.get('response', '')
        
        if not response_code:
            result = {
                'task_id': task_id,
                'test_result': 0,
                'failure_reason': 'No response_code found'
            }
            if progress_callback:
                progress_callback(task_id, "NO_CODE")
            return result
        
        # Validate and fix the code if needed (handles truncation issues)
        try:
            validated_code = validate_and_fix_code(response_code, response)
        except Exception as e:
            result = {
                'task_id': task_id,
                'test_result': 0,
                'failure_reason': f'Code validation error: {str(e)}'
            }
            if progress_callback:
                progress_callback(task_id, "VALIDATION_ERROR")
            return result
        
        # Convert function name to task_func (to match test expectations)
        converted_code = convert_function_to_task_func(response_code)
        
        # Get test code from dataset
        problem = problems[task_id]
        test_code = problem['test']
        
        # Execute and test
        result = safe_execute_code(converted_code, test_code, task_id)
        
        status = 'PASS' if result['test_result'] == 1 else 'FAIL'
        if progress_callback:
            progress_callback(task_id, status)
        
        return result
        
    except Exception as e:
        result = {
            'task_id': f'unknown_line_{line_idx}',
            'test_result': 0,
            'failure_reason': f'Processing error: {str(e)}'
        }
        if progress_callback:
            progress_callback(f'line_{line_idx}', f"ERROR: {str(e)[:30]}")
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

def process_generation_file_concurrent(file_path: str, problems: Dict[str, Any], max_workers: int = 8) -> List[Dict[str, Any]]:
    """Process a generation file and test each solution concurrently."""
    
    # Read all lines first
    with open(file_path, 'r') as f:
        lines = [(line.strip(), idx) for idx, line in enumerate(f) if line.strip()]
    
    results = []
    completed_count = 0
    total_lines = len(lines)
    
    # Create progress bar
    pbar = tqdm(total=total_lines, desc=f"Processing {Path(file_path).name}")
    
    # Thread-safe progress update
    progress_lock = threading.Lock()
    
    def update_progress(task_id: str, status: str):
        nonlocal completed_count
        with progress_lock:
            pbar.set_postfix_str(f"{task_id}: {status}")
            pbar.update(1)
    
    # Create a partial function with the problems and progress callback
    process_func = partial(process_single_sample, problems=problems, progress_callback=update_progress)
    
    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_line = {executor.submit(process_func, line_data): line_data for line_data in lines}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_line):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                line_data = future_to_line[future]
                error_result = {
                    'task_id': f'unknown_line_{line_data[1]}',
                    'test_result': 0,
                    'failure_reason': f'Future execution error: {str(e)}'
                }
                results.append(error_result)
    
    pbar.close()
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test generated code against BigCodeBench with concurrent execution')
    parser.add_argument('--model', type=str, required=True, 
                       help='Model name to process (e.g., o4, deepseek, gemini)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of concurrent workers (default: 8)')
    parser.add_argument('--pattern', type=str, default='code_*.jsonl',
                       help='File pattern to process (default: code_*.jsonl)')
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading BigCodeBench dataset...")
    problems = load_bigcodebench_dataset()
    print(f"Loaded {len(problems)} problems")
    
    # Process specified model
    generation_root = Path("/Users/aliredaq/Downloads/bigcodebench/generation_output")
    model_folder = generation_root / args.model
    
    if not model_folder.is_dir():
        print(f"Error: Model folder '{args.model}' not found in {generation_root}")
        print("Available models:")
        for folder in generation_root.iterdir():
            if folder.is_dir():
                print(f"  - {folder.name}")
        return
    
    print(f"\nProcessing model: {model_folder.name}")
    print(f"Using {args.workers} concurrent workers")
    print(f"Processing files matching: {args.pattern}")
    
    # Create test_generation folder
    test_gen_folder = model_folder / "test_generation"
    test_gen_folder.mkdir(exist_ok=True)
    
    # Process code generation files matching the pattern
    matching_files = list(model_folder.glob(args.pattern))
    if not matching_files:
        print(f"No files found matching pattern '{args.pattern}' in {model_folder}")
        return
    
    for gen_file in matching_files:
        print(f"  Processing {gen_file.name} with concurrent execution...")
        
        try:
            # Use concurrent processing
            results = process_generation_file_concurrent(str(gen_file), problems, max_workers=args.workers)
            
            # Save results
            output_file = test_gen_folder / f"test_{gen_file.name}"
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            # Print summary
            passed = sum(1 for r in results if r['test_result'] == 1)
            total = len(results)
            print(f"    Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
            
            # Print some failure examples for debugging
            failed_results = [r for r in results if r['test_result'] == 0]
            if failed_results:
                print(f"    Sample failures:")
                for i, fail in enumerate(failed_results[:3]):  # Show first 3 failures
                    reason = fail['failure_reason'][:100] + "..." if len(fail['failure_reason']) > 100 else fail['failure_reason']
                    print(f"      {fail['task_id']}: {reason}")
            
        except Exception as e:
            print(f"    Error processing {gen_file.name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()