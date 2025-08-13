#!/usr/bin/env python3
"""
Pipeline for cross-technique testing with model name parameter.
Includes standardization and cross-technique evaluation steps.
Uses proper temp folder sandboxing like test_func_evaluator.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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

def convert_function_to_task_func(code: str) -> str:
    """Convert the main function name to task_func."""
    original_name = extract_function_name(code)
    if not original_name:
        return code
    
    # Replace function definition
    pattern = rf'\bdef\s+{re.escape(original_name)}\s*\('
    replacement = 'def task_func('
    converted_code = re.sub(pattern, replacement, code)
    
    return converted_code

def remove_function_definitions(test_code: str) -> str:
    """Remove function definitions from test code, keeping only imports and test classes."""
    lines = test_code.split('\n')
    cleaned_lines = []
    in_function_def = False
    function_indent = 0
    
    for line in lines:
        line_stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())
        
        # Check if this is a function definition
        if line_stripped.startswith('def '):
            # Extract function name more carefully
            function_name = line_stripped.split('(')[0].replace('def ', '').strip()
            
            # Skip unittest test methods (they start with 'test_' and are indented)
            if function_name.startswith('test_') and current_indent > 0:
                # This is a unittest test method, keep it
                pass
            else:
                # These are functions under test that should be removed
                # Be more aggressive about removing any standalone function definitions
                if function_name in ['task_func', 'test_func', 'func', 'solution', 'my_function', 'your_function'] or \
                   any(name in function_name.lower() for name in ['task_func', 'test_func', 'solution', 'my_function', 'your_function']) or \
                   current_indent == 0:  # Top-level function definitions should be removed
                    in_function_def = True
                    function_indent = current_indent
                    continue
        
        # If we're inside a function definition, skip lines until we're back to the original indent level or less
        if in_function_def:
            if line_stripped == '':  # Empty line
                continue
            elif current_indent > function_indent:  # Still inside function
                continue
            else:  # Back to original indent level or less, function definition ended
                in_function_def = False
                function_indent = 0
                # Don't continue here, we want to process this line normally
        
        # Keep the line if we're not in a function definition
        if not in_function_def:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def clean_test_imports(test_code: str) -> str:
    """Clean problematic imports from test code."""
    lines = test_code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip problematic imports
        if (line_stripped.startswith('import ') or line_stripped.startswith('from ')) and \
           any(x in line_stripped for x in [
               'your_module', 'my_module', 'task', 'solution', 'module', 
               'task_module', 'your_solution', 'my_solution', 'test_module',
               'from task ', 'import task', 'task import', 'solution import',
               'your_', 'my_', 'module_', 'task_', 'solution_',
               'import your', 'import my', 'import task', 'import solution',
               'from your', 'from my', 'from task', 'from solution'
           ]):
            # Skip this line completely
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def convert_test_calls_to_task_func(test_code: str) -> str:
    """Convert function calls in test code to use task_func."""
    # Define built-in functions that should NOT be replaced
    builtin_functions = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
        'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
        'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
        'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
        'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
        'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
        'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
        'super', 'tuple', 'type', 'vars', 'zip', '__import__',
        # Common test methods that should not be replaced
        'setUp', 'tearDown', 'assertEqual', 'assertTrue', 'assertFalse', 
        'assertRaises', 'assertIn', 'assertNotIn', 'mock', 'patch', 'call'
    }
    
    # Replace various function call patterns with task_func
    patterns = [
        (r'\btest_func\s*\(', 'task_func('),
        (r'\byour_function\s*\(', 'task_func('),
        (r'\bmy_function\s*\(', 'task_func('),
        (r'\bsolution\s*\(', 'task_func('),
        (r'\btask\s*\(', 'task_func('),
        (r'\bfunc\s*\(', 'task_func('),  # This is important - "func" is often the function under test
    ]
    
    for pattern, replacement in patterns:
        # First, find all matches
        matches = re.finditer(pattern, test_code)
        
        # Process matches in reverse order to avoid index issues when replacing
        for match in reversed(list(matches)):
            # Extract the function name being called
            func_name = match.group(0).replace('(', '').strip()
            
            # Skip if it's a built-in function or common test method
            if func_name not in builtin_functions:
                # Check if this is not a method call (no dot before it)
                start_pos = match.start()
                if start_pos == 0 or test_code[start_pos - 1] not in '.':
                    test_code = test_code[:match.start()] + replacement + test_code[match.end():]
    
    return test_code

def safe_execute_code(code: str, test_code: str, task_id: str, code_technique: str, test_technique: str) -> Dict[str, Any]:
    """Safely execute code with test in a temporary file using test_environment."""
    result = {
        'task_id': task_id,
        'code_technique': code_technique,
        'test_technique': test_technique,
        'test_results': [],
        'pass_rate': 0.0,
        'failure_reason': None,
        'executed_code': None,
        'executed_test': None,
        'detailed_error': None
    }
    
    # Path to test environment python
    test_env_python = "/Users/aliredaq/Downloads/bigcodebench/test_environment/bin/python"
    
    try:
        # Store the executed code and test for debugging
        result['executed_code'] = code
        result['executed_test'] = test_code
        
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
                # Parse test results from stdout
                test_results, failure_reason = parse_test_output(process.stdout, process.stderr, process.returncode, test_code)
                result['test_results'] = test_results
                result['pass_rate'] = sum(int(r) for r in test_results) / len(test_results) if test_results else 0.0
                result['failure_reason'] = failure_reason
            else:
                # Parse test results from stderr
                test_results, failure_reason = parse_test_output(process.stdout, process.stderr, process.returncode, test_code)
                result['test_results'] = test_results if test_results else ["0"]
                result['pass_rate'] = sum(int(r) for r in result['test_results']) / len(result['test_results']) if result['test_results'] else 0.0
                result['failure_reason'] = failure_reason
                result['detailed_error'] = failure_reason
                
        except subprocess.TimeoutExpired:
            result['test_results'] = ["0"]
            result['pass_rate'] = 0.0
            result['failure_reason'] = "Timeout: Code execution exceeded 60 seconds"
            result['detailed_error'] = "Timeout: Code execution exceeded 60 seconds"
            
        except Exception as e:
            result['test_results'] = ["0"]
            result['pass_rate'] = 0.0
            result['failure_reason'] = f"Execution error: {str(e)}"
            result['detailed_error'] = f"Execution error: {str(e)}"
            
    except Exception as e:
        result['test_results'] = ["0"]
        result['pass_rate'] = 0.0
        result['failure_reason'] = f"Setup error: {str(e)}"
        result['detailed_error'] = f"Setup error: {str(e)}"
        
    finally:
        # Clean up temp file
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except:
            pass
    
    return result

def count_test_methods_in_code(test_code: str) -> int:
    """Count the number of test methods in the test code."""
    if not test_code:
        return 1
    
    # Count test methods by looking for "def test_" pattern
    test_method_count = len(re.findall(r'def\s+test_\w+', test_code))
    
    # If no test methods found, assume 1
    return max(1, test_method_count)

def parse_test_output(stdout: str, stderr: str, returncode: int, test_code: str = "") -> Tuple[List[str], str]:
    """Parse unittest output to extract individual test results."""
    test_results = []
    failure_reason = None
    combined_output = stdout + "\n" + stderr
    
    # First, look for the unittest result line with dots, F, E pattern
    # This is the most reliable way to get individual test results
    result_pattern = re.search(r'^([.FE]+)$', combined_output, re.MULTILINE)
    
    if result_pattern:
        # Found the unittest result line (e.g., "...F.E.")
        result_string = result_pattern.group(1)
        for char in result_string:
            if char == '.':
                test_results.append("1")  # Passed
            elif char in 'FE':
                test_results.append("0")  # Failed or Error
    else:
        # Fallback 1: Look for individual test method results
        test_method_names = []
        
        # Look for test method executions in various formats
        for line in combined_output.split('\n'):
            # Pattern 1: test_method_name (TestClass) ... ok/FAIL/ERROR
            if 'test_' in line and ('...' in line or 'ok' in line or 'FAIL' in line or 'ERROR' in line):
                test_method_names.append(line.strip())
            # Pattern 2: TestClass.test_method_name ... ok/FAIL/ERROR
            elif '.' in line and 'test_' in line and ('ok' in line or 'FAIL' in line or 'ERROR' in line):
                test_method_names.append(line.strip())
        
        # If we found individual test results, parse them
        if test_method_names:
            for test_line in test_method_names:
                if 'ok' in test_line.lower() or '... ok' in test_line:
                    test_results.append("1")  # Passed
                elif 'FAIL' in test_line or 'ERROR' in test_line or 'fail' in test_line.lower():
                    test_results.append("0")  # Failed
                else:
                    # Default to failed if we can't determine
                    test_results.append("0")
        else:
            # Fallback 2: Use "Ran X tests" pattern
            ran_pattern = re.search(r'Ran (\d+) tests?', combined_output)
            if ran_pattern:
                test_count = int(ran_pattern.group(1))
                if returncode == 0:
                    # All tests passed
                    test_results = ["1"] * test_count
                else:
                    # Some tests failed, try to determine how many
                    # Look for specific failure patterns
                    fail_patterns = [
                        r'FAIL: test_\w+',
                        r'ERROR: test_\w+',
                        r'test_\w+ \(.*\) ... FAIL',
                        r'test_\w+ \(.*\) ... ERROR'
                    ]
                    
                    failed_tests = set()
                    for pattern in fail_patterns:
                        failed_tests.update(re.findall(pattern, combined_output))
                    
                    failed_count = len(failed_tests)
                    passed_count = max(0, test_count - failed_count)
                    test_results = ["1"] * passed_count + ["0"] * failed_count
            else:
                # Fallback 3: Count test methods from the test code itself
                test_count = count_test_methods_in_code(test_code)
                if returncode == 0:
                    # All tests passed
                    test_results = ["1"] * test_count
                else:
                    # Some tests failed, but we can't determine exact count
                    # Look for any indication of failures
                    if 'FAILED' in combined_output or 'ERROR' in combined_output:
                        # Assume half failed (rough estimate)
                        failed_count = max(1, test_count // 2)
                        passed_count = test_count - failed_count
                        test_results = ["1"] * passed_count + ["0"] * failed_count
                    else:
                        # All tests likely failed
                        test_results = ["0"] * test_count
    
    # Extract failure reason with more details
    if returncode != 0:
        if 'TypeError' in combined_output and ('takes' in combined_output and 'positional argument' in combined_output):
            failure_reason = "Function signature mismatch (different parameter counts)"
            # Try to find the specific error line
            for line in combined_output.split('\n'):
                if 'TypeError' in line and 'takes' in line:
                    failure_reason += f" - {line.strip()}"
                    break
        elif 'TypeError' in combined_output:
            failure_reason = "Argument type mismatch"
            # Find the specific TypeError
            for line in combined_output.split('\n'):
                if 'TypeError' in line:
                    failure_reason += f" - {line.strip()}"
                    break
        elif 'AssertionError' in combined_output:
            failure_reason = "Test assertion failures"
            # Find assertion details
            for line in combined_output.split('\n'):
                if 'AssertionError' in line or ('assert' in line.lower() and 'fail' in line.lower()):
                    failure_reason += f" - {line.strip()}"
                    break
        elif 'FAILED' in combined_output:
            failure_reason = "Test assertion failures"
        elif 'ERROR' in combined_output:
            failure_reason = "Test execution errors"
        elif 'ModuleNotFoundError' in combined_output:
            failure_reason = "Missing module dependency"
            # Find the specific module
            for line in combined_output.split('\n'):
                if 'ModuleNotFoundError' in line:
                    failure_reason += f" - {line.strip()}"
                    break
        elif 'SyntaxError' in combined_output:
            failure_reason = "Syntax error in code"
            # Find the syntax error details
            for line in combined_output.split('\n'):
                if 'SyntaxError' in line:
                    failure_reason += f" - {line.strip()}"
                    break
        elif 'NameError' in combined_output:
            failure_reason = "Name error - function not defined correctly"
            # Find the specific name error
            for line in combined_output.split('\n'):
                if 'NameError' in line:
                    failure_reason += f" - {line.strip()}"
                    break
        else:
            failure_reason = f"Unknown test failure - Combined output: {combined_output[:500]}..."
    
    # Ensure we have at least one result
    if not test_results:
        test_results = ["0"]  # Default to failed if we can't parse
    
    return test_results, failure_reason


class CrossTechniquePipeline:
    """Pipeline for cross-technique testing with model name parameter."""
    
    def __init__(self, model_name: str, base_dir: str = "generation_output"):
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / model_name
        self.code_standardized_dir = self.base_dir / f"{model_name}_standardized"
        self.test_standardized_dir = self.base_dir / f"{model_name}_test_standardized"
        self.results_dir = self.base_dir / model_name / "cross_result"
        
    def validate_directories(self) -> bool:
        """Validate that required directories exist."""
        if not self.model_dir.exists():
            print(f"Error: Model directory {self.model_dir} does not exist!")
            return False
        
        # Check if we have both code and test files
        code_files = list(self.model_dir.glob("code_*.jsonl"))
        test_files = list(self.model_dir.glob("test_*.jsonl"))
        
        if not code_files:
            print(f"Error: No code_*.jsonl files found in {self.model_dir}")
            return False
            
        if not test_files:
            print(f"Error: No test_*.jsonl files found in {self.model_dir}")
            return False
            
        print(f"Found {len(code_files)} code files and {len(test_files)} test files in {self.model_dir}")
        return True
    
    def run_code_standardization(self) -> bool:
        """Run the code standardization process."""
        print("Step 1: Standardizing code function names...")
        
        try:
            result = subprocess.run([
                sys.executable, 
                "standardize_function_names.py",
                "--input-dir", str(self.model_dir),
                "--output-dir", str(self.code_standardized_dir)
            ], capture_output=True, text=True, check=True)
            
            print("Code function name standardization completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error in code function name standardization: {e}")
            print(f"stderr: {e.stderr}")
            return False
    
    def run_test_standardization(self) -> bool:
        """Run the test standardization process."""
        print("Step 2: Standardizing test function names...")
        
        try:
            result = subprocess.run([
                sys.executable, 
                "standardize_test_function_names.py",
                "--directory", str(self.model_dir),
                "--output-dir", str(self.test_standardized_dir)
            ], capture_output=True, text=True, check=True)
            
            print("Test function name standardization completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error in test function name standardization: {e}")
            print(f"stderr: {e.stderr}")
            return False
    
    def load_jsonl_file(self, file_path: Path) -> Dict[str, Dict]:
        """Load JSONL file and return as dictionary keyed by task_id."""
        data = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        task_id = entry['task_id']
                        data[task_id] = entry
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON on line {line_num} in {file_path}: {e}")
                        continue
        return data
    
    def extract_technique_name(self, filename: str) -> str:
        """Extract technique name from filename."""
        # Remove code_ or test_ prefix and file extension
        name = filename.replace('code_', '').replace('test_', '')
        
        # Handle different formats:
        # Current format: code_{technique}_{model}_bigcodebench_output_llm{X}[_v{Y}].jsonl
        # Bigcode format: code_{technique}_{model}_bigcodebench_v{X}.jsonl
        
        # Split by underscore and find the technique (first part after code_/test_)
        parts = name.split('_')
        if len(parts) > 0:
            # First part is the technique name
            technique = parts[0]
            return technique
        
        # Fallback: remove common suffixes
        name = name.replace('.jsonl', '')
        for suffix in ['_bigcodebench_output_llm1_v2', '_bigcodebench_output_llm1', '_bigcodebench_v1', '_bigcodebench_v2', '_bigcodebench_v3', '_bigcodebench_v4', '_bigcodebench_v5', '_bigcodebench_v6', '_bigcodebench_v7', '_bigcodebench_v8']:
            if suffix in name:
                name = name.replace(suffix, '')
                break
        
        return name
        
    def process_cross_technique_pair(self, code_file: Path, test_file: Path) -> List[Dict[str, Any]]:
        """Process a cross-technique pair with proper sandboxing."""
        # Extract technique names using the improved method
        code_technique = self.extract_technique_name(code_file.name)
        test_technique = self.extract_technique_name(test_file.name)
        
        # Load code and test data
        code_data = self.load_jsonl_file(code_file)
        test_data = self.load_jsonl_file(test_file)
        
        # Find common task IDs
        common_tasks = set(code_data.keys()) & set(test_data.keys())
        
        results = []
        
        # Process each task with progress bar
        with tqdm(total=len(common_tasks), desc=f"    {code_file.name} vs {test_file.name}", leave=False) as pbar:
            for task_id in sorted(common_tasks):
                code_entry = code_data[task_id]
                test_entry = test_data[task_id]
                
                # Get the generated code and test
                generated_code = code_entry.get('response_code', '')
                generated_test = test_entry.get('response_code', '')
                
                if not generated_code or not generated_test:
                    results.append({
                        'task_id': task_id,
                        'code_technique': code_technique,
                        'test_technique': test_technique,
                        'test_results': ["0"],
                        'pass_rate': 0.0,
                        'failure_reason': 'Missing code or test',
                        'executed_code': None,
                        'executed_test': None,
                        'detailed_error': 'Missing code or test'
                    })
                    pbar.set_postfix_str(f"{task_id}: Missing data")
                    pbar.update(1)
                    continue
                
                # Convert function name to task_func
                converted_code = convert_function_to_task_func(generated_code)
                
                # Clean test code: remove function definitions, clean imports, and convert calls
                test_without_functions = remove_function_definitions(generated_test)
                cleaned_test = clean_test_imports(test_without_functions)
                converted_test = convert_test_calls_to_task_func(cleaned_test)
                
                # Execute and test using generated test (not original test)
                result = safe_execute_code(converted_code, converted_test, task_id, code_technique, test_technique)
                results.append(result)
                
                status = 'PASS' if result['pass_rate'] > 0 else 'FAIL'
                pbar.set_postfix_str(f"{task_id}: {status}")
                pbar.update(1)
        
        return results
    
    def run_cross_technique_testing(self) -> bool:
        """Run the cross-technique testing process with proper sandboxing and progress bars."""
        print("Step 3: Running cross-technique testing with temp folder sandboxing...")
        
        try:
            # Get all code and test files
            code_files = list(self.model_dir.glob("code_*.jsonl"))
            test_files = list(self.model_dir.glob("test_*.jsonl"))
            
            if not code_files or not test_files:
                print("No code or test files found")
                return False
            
            print(f"Found {len(code_files)} code files and {len(test_files)} test files")
            print(f"Code techniques: {[self.extract_technique_name(f.name) for f in code_files]}")
            print(f"Test techniques: {[self.extract_technique_name(f.name) for f in test_files]}")
            print(f"Will process {len(code_files)} × {len(test_files)} = {len(code_files) * len(test_files)} combinations")
            
            # Ensure results directory exists
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all combinations with progress bar
            all_summaries = []
            total_combinations = len(code_files) * len(test_files)
            
            with tqdm(total=total_combinations, desc="Cross-technique combinations", unit="combo") as pbar:
                for code_file in code_files:
                    for test_file in test_files:
                        # Extract technique names for output file
                        code_technique = self.extract_technique_name(code_file.name)
                        test_technique = self.extract_technique_name(test_file.name)
                        
                        pbar.set_description(f"Processing {code_technique} vs {test_technique}")
                        
                        # Process this pair
                        results = self.process_cross_technique_pair(code_file, test_file)
                        
                        # Save results
                        output_file = self.results_dir / f"code_{code_technique}_test_{test_technique}.jsonl"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for result in results:
                                f.write(json.dumps(result) + '\n')
                        
                        # Calculate summary
                        total_tasks = len(results)
                        total_tests = sum(len(r['test_results']) for r in results)
                        total_passed = sum(sum(int(tr) for tr in r['test_results']) for r in results)
                        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
                        
                        summary = {
                            'code_technique': code_technique,
                            'test_technique': test_technique,
                            'total_tasks': total_tasks,
                            'total_tests': total_tests,
                            'total_passed': total_passed,
                            'overall_pass_rate': overall_pass_rate
                        }
                        all_summaries.append(summary)
                        
                        # Update progress bar with current results
                        pbar.update(1)
                        pbar.set_postfix({
                            'pass_rate': f"{overall_pass_rate:.1%}",
                            'tests': f"{total_passed}/{total_tests}"
                        })
            
            # Save comprehensive summary
            summary_file = self.results_dir / "comprehensive_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_name': self.model_name,
                    'total_combinations': len(all_summaries),
                    'results': all_summaries
                }, f, indent=2)
            
            print("Cross-technique testing completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error in cross-technique testing: {e}")
            traceback.print_exc()
            return False
    
    def create_model_specific_results(self) -> bool:
        """Create model-specific results with renamed files."""
        print("Step 4: Creating model-specific results...")
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Rename result files to include model name
        if self.results_dir.exists():
            for result_file in self.results_dir.glob("code_*.jsonl"):
                # Check if model name is already in filename
                if f"_{self.model_name}_" not in result_file.name:
                    # Insert model name into filename
                    parts = result_file.name.split('_')
                    if len(parts) >= 4:  # code_X_test_Y.jsonl
                        new_name = f"code_{self.model_name}_{parts[1]}_test_{self.model_name}_{parts[3]}"
                        new_path = result_file.parent / new_name
                        
                        # Rename the file
                        result_file.rename(new_path)
                        print(f"  Renamed: {result_file.name} -> {new_name}")
            
            # Rename summary file if it exists
            summary_file = self.results_dir / "fixed_comprehensive_summary.json"
            if summary_file.exists():
                model_summary = self.results_dir / f"{self.model_name}_comprehensive_summary.json"
                if summary_file != model_summary:
                    summary_file.rename(model_summary)
                    print(f"  Renamed: fixed_comprehensive_summary.json -> {model_summary.name}")
        
        return True
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline with progress tracking."""
        print(f"{'='*60}")
        print(f"Running Cross-Technique Pipeline for model: {self.model_name}")
        print(f"{'='*60}")
        
        if not self.validate_directories():
            return False
        
        # Overall pipeline progress
        total_steps = 4
        with tqdm(total=total_steps, desc="Pipeline Progress", unit="step") as pipeline_pbar:
            # Step 1: Standardize code function names
            pipeline_pbar.set_description("Step 1/4: Code Standardization")
            with tqdm(desc="Standardizing code function names", leave=False) as step_pbar:
                if not self.run_code_standardization():
                    return False
                step_pbar.update(1)
            pipeline_pbar.update(1)
            
            # Step 2: Standardize test function names
            pipeline_pbar.set_description("Step 2/4: Test Standardization")
            with tqdm(desc="Standardizing test function names", leave=False) as step_pbar:
                if not self.run_test_standardization():
                    return False
                step_pbar.update(1)
            pipeline_pbar.update(1)
            
            # Step 3: Run cross-technique testing
            pipeline_pbar.set_description("Step 3/4: Cross-Technique Testing")
            if not self.run_cross_technique_testing():
                return False
            pipeline_pbar.update(1)
            
            # Step 4: Create model-specific results
            pipeline_pbar.set_description("Step 4/4: Organizing Results")
            with tqdm(desc="Creating model-specific results", leave=False) as step_pbar:
                if not self.create_model_specific_results():
                    return False
                step_pbar.update(1)
            pipeline_pbar.update(1)
            
            pipeline_pbar.set_description("Pipeline Completed")
        
        print(f"\n{'='*60}")
        print("Pipeline completed successfully!")
        print(f"Results saved to: {self.results_dir}/")
        print(f"{'='*60}")
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cross-Technique Testing Pipeline")
    parser.add_argument("--model", required=True, 
                       help="Model name (e.g., 'o4-mini')")
    parser.add_argument("--base-dir", default="generation_output",
                       help="Base directory containing model outputs")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = CrossTechniquePipeline(args.model, args.base_dir)
    success = pipeline.run_pipeline()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()