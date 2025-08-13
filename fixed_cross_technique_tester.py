#!/usr/bin/env python3
"""
Fixed Cross-technique testing framework for BigCodeBench.
Combines the original approach with safety improvements.
"""

import json
import os
import sys
import tempfile
import shutil
import subprocess
import venv
import threading
import time
import gc
import signal
import psutil
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def convert_code_function_to_task_func(code: str) -> str:
    """Convert the main function definition to task_func for code files."""
    original_name = extract_function_name(code)
    if not original_name:
        return code
    
    # Replace function definition
    pattern = rf'\bdef\s+{re.escape(original_name)}\s*\('
    replacement = 'def task_func('
    converted_code = re.sub(pattern, replacement, code)
    
    # Replace function calls within the same code
    call_pattern = rf'\b{re.escape(original_name)}\s*\('
    call_replacement = 'task_func('
    converted_code = re.sub(call_pattern, call_replacement, converted_code)
    
    return converted_code

def extract_function_under_test_from_test_code(test_code: str) -> Optional[str]:
    """Extract the function name being tested from test code."""
    # Look for function calls that are likely the function under test
    # Exclude common test framework and builtin functions
    excluded_functions = {
        'assertEqual', 'assertTrue', 'assertFalse', 'assertRaises', 'assertIn', 'assertNotIn',
        'assertIsNone', 'assertIsNotNone', 'assertGreater', 'assertLess', 'assertGreaterEqual',
        'assertLessEqual', 'assertAlmostEqual', 'assertNotAlmostEqual', 'assertListEqual',
        'assertDictEqual', 'assertSetEqual', 'assertTupleEqual', 'assertSequenceEqual',
        'assertMultiLineEqual', 'assertRegex', 'assertNotRegex', 'assertCountEqual',
        'fail', 'skipTest', 'subTest', 'addCleanup', 'setUp', 'tearDown', 'setUpClass',
        'tearDownClass', 'print', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict',
        'set', 'tuple', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
        'sum', 'min', 'max', 'abs', 'round', 'pow', 'divmod', 'isinstance', 'issubclass',
        'hasattr', 'getattr', 'setattr', 'delattr', 'open', 'close', 'read', 'write',
        'mock', 'patch', 'MagicMock', 'Mock', 'call', 'side_effect', 'return_value'
    }
    
    # Find all function calls in the test code
    function_calls = re.findall(r'\b(\w+)\s*\(', test_code)
    
    # Count frequency of each function call
    call_counts = {}
    for func_name in function_calls:
        if func_name not in excluded_functions and not func_name.startswith('test_'):
            call_counts[func_name] = call_counts.get(func_name, 0) + 1
    
    # Return the most frequently called non-excluded function
    if call_counts:
        return max(call_counts, key=call_counts.get)
    
    return None

def convert_test_function_calls_to_task_func(test_code: str) -> str:
    """Convert the function under test calls to task_func in test files."""
    # Extract the function being tested from the test code itself
    function_under_test = extract_function_under_test_from_test_code(test_code)
    
    if not function_under_test:
        return test_code
    
    # Only replace calls to the specific function under test
    # Don't replace def statements or third-party/builtin functions
    call_pattern = rf'\b{re.escape(function_under_test)}\s*\('
    call_replacement = 'task_func('
    converted_test = re.sub(call_pattern, call_replacement, test_code)
    
    return converted_test


class FixedCrossTechniqueTester:
    """Improved cross-technique tester that fixes the original issues."""
    
    def __init__(self, code_dir: str, test_dir: str, output_dir: str = "cross_technique_results", max_workers: int = 4):
        self.code_dir = Path(code_dir).resolve()
        self.test_dir = Path(test_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Threading configuration (safer than multiprocessing)
        self.max_workers = max_workers
        self.memory_threshold = 0.85
        
        # Thread safety for file operations
        self.file_lock = threading.Lock()
        # Additional lock for directory creation to prevent race conditions
        self.dir_lock = threading.Lock()
        
        # Setup test environment (like concurrent_test_func_evaluator)
        self.test_env_python = "/Users/aliredaq/Downloads/bigcodebench/test_environment/bin/python"
        
        if not Path(self.test_env_python).exists():
            print(f"Error: Test environment Python executable not found at {self.test_env_python}")
            print("Please ensure the test environment is set up")
            sys.exit(1)
    
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
        name = filename.replace('code_', '').replace('test_', '')
        name = name.replace('_o4-mini_bigcodebench_output_llm.jsonl', '')
        name = name.replace('.jsonl', '')
        return name
    
    def get_code_files(self) -> List[Path]:
        """Get all code_*.jsonl files."""
        return sorted(list(self.code_dir.glob("code_*.jsonl")))
    
    def get_test_files(self) -> List[Path]:
        """Get all test_*.jsonl files."""
        return sorted(list(self.test_dir.glob("test_*.jsonl")))
    
    def clean_test_code(self, test_code: str) -> str:
        """Clean test code by removing problematic imports and patches."""
        lines = test_code.split('\n')
        cleaned_lines = []
        skip_next_lines = 0
        patched_functions = []
        
        for i, line in enumerate(lines):
            if skip_next_lines > 0:
                skip_next_lines -= 1
                continue
                
            line_stripped = line.strip()
            
            # Skip problematic imports
            if (line_stripped.startswith('import ') or line_stripped.startswith('from ')) and \
               any(x in line_stripped for x in ['ftp_download', 'module', 'solution', 'your_module', 'task import task_func', 'task_module import', 'from task ']):
                continue
            
            # Skip @patch decorators that reference non-existent modules
            if line_stripped.startswith('@patch(') and any(x in line_stripped for x in ['ftp_download', 'module', 'solution']):
                # Also skip the next line (the function definition), but remember to clean its signature
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('def '):
                        # Extract function name and clean its signature
                        func_def = next_line
                        # Remove mock parameters from function signature
                        import re
                        # Remove parameters that look like mock_* or *_mock
                        func_def = re.sub(r',\s*mock_\w+', '', func_def)
                        func_def = re.sub(r',\s*\w+_mock', '', func_def)
                        # Handle case where mock parameter is first
                        func_def = re.sub(r'\(mock_\w+,\s*', '(', func_def)
                        func_def = re.sub(r'\(\w+_mock,\s*', '(', func_def)
                        # Handle case where mock parameter is only parameter
                        func_def = re.sub(r'\(mock_\w+\)', '(self)', func_def)
                        func_def = re.sub(r'\(\w+_mock\)', '(self)', func_def)
                        
                        lines[i + 1] = func_def
                
                skip_next_lines = 1
                continue
            
            # Fix test_func calls to use task_func instead
            if 'test_func(' in line:
                line = line.replace('test_func(', 'task_func(')
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def prepare_combined_test_file(self, temp_dir: Path, code_entry: Dict, test_entry: Dict) -> Path:
        """Prepare a complete test file combining generated code and generated tests."""
        test_file = temp_dir / "combined_test.py"
        
        # Get the generated code
        generated_code = code_entry.get('response_code', '')
        if not generated_code:
            raise ValueError("No generated code found")
        
        # Convert code function to task_func
        converted_code = convert_code_function_to_task_func(generated_code)
        
        # Get and clean the generated test
        generated_test = test_entry.get('response_code', '')
        if not generated_test:
            raise ValueError("No generated test found")
        
        # Clean the test code
        cleaned_test = self.clean_test_code(generated_test)
        
        # Convert function under test calls to task_func in test
        cleaned_test = convert_test_function_calls_to_task_func(cleaned_test)
        
        # Extract imports from both code and test
        imports = set()
        
        # Extract imports from converted code
        for line in converted_code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.add(line)
        
        # Extract imports from cleaned test
        for line in cleaned_test.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Skip already filtered problematic imports
                imports.add(line)
        
        # Prepare the complete test file content
        test_content = []
        
        # Add matplotlib configuration first
        test_content.append('# Matplotlib configuration')
        test_content.append('import matplotlib')
        test_content.append('matplotlib.use("Agg")')
        test_content.append('')
        
        # Add all unique imports (excluding matplotlib ones we already added)
        filtered_imports = [imp for imp in sorted(imports) if 'matplotlib' not in imp]
        test_content.extend(filtered_imports)
        test_content.append('')
        
        # Add converted code (this defines task_func)
        test_content.append('# Generated code:')
        test_content.append(converted_code)
        test_content.append('')
        
        # Create alias from task_func to test_func for compatibility
        test_content.append('# Function alias for test compatibility:')
        test_content.append('try:')
        test_content.append('    test_func = task_func')
        test_content.append('except NameError:')
        test_content.append('    pass  # task_func not defined')
        test_content.append('')
        
        # Add cleaned test
        test_content.append('# Generated tests (cleaned):')
        test_content.append(cleaned_test)
        test_content.append('')
        
        # Ensure unittest.main() is present
        if 'unittest.main()' not in cleaned_test:
            test_content.append('if __name__ == "__main__":')
            test_content.append('    unittest.main()')
        
        # Write test file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_content))
        
        return test_file
    
    def check_system_resources(self) -> bool:
        """Check if system has enough resources."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent < self.memory_threshold
        except:
            return True
    
    def run_test_in_env(self, temp_dir: Path, test_file: Path) -> Tuple[bool, List[str], str]:
        """Run the test file and return results."""
        original_cwd = os.getcwd()
        try:
            # Change to temp directory for test execution
            os.chdir(temp_dir)
            
            # Run the test using the test environment
            result = subprocess.run(
                [str(self.test_env_python), str(test_file), '-v'],
                capture_output=True,
                text=True,
                timeout=60  # Match concurrent_test_func_evaluator timeout
            )
            
            # Parse test results
            test_results, failure_reason = self.parse_test_output(result.stdout, result.stderr, result.returncode)
            success = result.returncode == 0
            
            return success, test_results, failure_reason
            
        except subprocess.TimeoutExpired:
            return False, [], "Test timed out after 60 seconds"
        except Exception as e:
            return False, [], f"Error running test: {str(e)}"
        finally:
            # Always restore original directory
            os.chdir(original_cwd)
    
    def parse_test_output(self, stdout: str, stderr: str, returncode: int) -> Tuple[List[str], str]:
        """Parse unittest output to extract individual test results."""
        test_results = []
        failure_reason = None
        
        if returncode == 0:
            # All tests passed - count the number of test methods
            lines = stdout.split('\n')
            for line in lines:
                if 'test_' in line and ('ok' in line.lower() or '...' in line):
                    test_results.append("1")  # Passed
            
            # If we couldn't parse individual results, assume all passed
            if not test_results:
                # Count test methods in a different way
                test_count = stdout.count('test_') if 'test_' in stdout else 1
                test_results = ["1"] * max(1, test_count)
        else:
            # Some tests failed - parse detailed output
            combined_output = stdout + "\n" + stderr
            
            # Look for test method results
            lines = combined_output.split('\n')
            for line in lines:
                if 'test_' in line:
                    if 'ok' in line.lower() or 'PASS' in line:
                        test_results.append("1")  # Passed
                    elif 'FAIL' in line or 'ERROR' in line or 'fail' in line.lower():
                        test_results.append("0")  # Failed
            
            # Extract failure reason with more details
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
    
    def check_compatibility(self, code: str, test: str) -> bool:
        """Check if code and test are potentially compatible."""
        # Basic checks first
        if not code or not test:
            return False
        
        # Check if test contains requests for specification instead of actual tests
        invalid_phrases = [
            "Could you please provide",
            "I need the specification",
            "provide the specification",
            "I'll need the specification",
            "description of what",
            "what `test_func` is supposed to do"
        ]
        
        if any(phrase in test for phrase in invalid_phrases):
            return False
        
        # Check if code has a function definition
        if 'def task_func(' not in code:
            return False
        
        # Check if test has actual test methods and calls to task_func
        if 'class Test' not in test or 'task_func(' not in test:
            return False
        
        # If we get here, it's likely a valid code-test pair
        # Let the test execution handle any runtime incompatibilities
        return True

    def test_code_against_test(self, code_entry: Dict, test_entry: Dict, 
                              code_technique: str, test_technique: str) -> Dict:
        """Test a single code entry against a single test entry."""
        task_id = code_entry['task_id']
        
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
        
        try:
            # Check system resources before proceeding
            if not self.check_system_resources():
                time.sleep(0.5)
            
            # Get the code and test first for compatibility check
            code = code_entry.get('response_code', '')
            test = test_entry.get('response_code', '')
            
            if not code or not test:
                result['test_results'] = ["0", "0", "0", "0"]
                result['pass_rate'] = 0.0
                result['failure_reason'] = "Missing code or test"
                return result
            
            # Check compatibility before testing
            converted_code = convert_code_function_to_task_func(code)
            cleaned_test = self.clean_test_code(test)
            converted_test = convert_test_function_calls_to_task_func(cleaned_test)
            
            # Store the modified code and test for debugging
            result['executed_code'] = converted_code
            result['executed_test'] = converted_test
            
            if not self.check_compatibility(converted_code, converted_test):
                result['test_results'] = ["0"]
                result['pass_rate'] = 0.0
                result['failure_reason'] = "Invalid test code (likely specification request)"
                result['detailed_error'] = f"Test appears to be a specification request rather than actual test code. Test snippet: {converted_test[:200]}..."
                return result
            
            # Create temporary directory for this specific test with thread safety
            with tempfile.TemporaryDirectory() as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                
                try:
                    # Prepare combined test file
                    test_file = self.prepare_combined_test_file(temp_dir, code_entry, test_entry)
                    
                    # Run test
                    success, test_results, failure_reason = self.run_test_in_env(temp_dir, test_file)
                    
                    result['test_results'] = test_results
                    result['pass_rate'] = sum(int(r) for r in test_results) / len(test_results) if test_results else 0.0
                    result['failure_reason'] = failure_reason if not success else None
                    
                    # Store detailed error information
                    if not success:
                        result['detailed_error'] = failure_reason
                        # If we have specific error output, include more details
                        if failure_reason and "error" in failure_reason.lower():
                            # Try to extract the most relevant error line
                            stderr_lines = failure_reason.split('\n') if '\n' in failure_reason else [failure_reason]
                            # Find the actual error line (usually the last substantive line)
                            for line in reversed(stderr_lines):
                                if any(err_type in line for err_type in ['TypeError', 'ValueError', 'AttributeError', 'NameError', 'ImportError']):
                                    result['detailed_error'] = line.strip()
                                    break
                    
                except Exception as e:
                    result['test_results'] = ["0"]
                    result['pass_rate'] = 0.0
                    result['failure_reason'] = f"Framework error: {str(e)}"
                finally:
                    # Force garbage collection to free memory
                    gc.collect()
        
        except Exception as e:
            result['test_results'] = ["0"]
            result['pass_rate'] = 0.0
            result['failure_reason'] = f"Outer framework error: {str(e)}"
        
        return result
    
    def test_technique_pair_worker(self, args) -> List[Dict]:
        """Worker function for testing a technique pair."""
        code_file, test_file = args
        code_technique = self.extract_technique_name(code_file.name)
        test_technique = self.extract_technique_name(test_file.name)
        
        # Load data
        code_data = self.load_jsonl_file(code_file)
        test_data = self.load_jsonl_file(test_file)
        
        results = []
        
        # Find common task IDs
        common_tasks = set(code_data.keys()) & set(test_data.keys())
        
        # Process tasks with progress bar
        with tqdm(total=len(common_tasks), desc=f"  {code_technique} vs {test_technique}", leave=False) as task_pbar:
            for task_id in sorted(common_tasks):
                if not self.check_system_resources():
                    task_pbar.set_postfix_str(f"Skipping {task_id} - insufficient resources")
                    task_pbar.update(1)
                    continue
                    
                code_entry = code_data[task_id]
                test_entry = test_data[task_id]
                
                task_pbar.set_postfix_str(f"Testing {task_id}")
                
                result = self.test_code_against_test(code_entry, test_entry, code_technique, test_technique)
                results.append(result)
                
                # Brief pause between tests
                time.sleep(0.1)
                task_pbar.update(1)
        
        return results
    
    def save_results(self, results: List[Dict], code_technique: str, test_technique: str) -> Dict:
        """Save results for a technique pair."""
        filename = f"code_{code_technique}_test_{test_technique}.jsonl"
        output_file = self.output_dir / filename
        
        # Use lock for thread-safe file operations
        with self.file_lock:
            # Ensure output directory exists with directory lock
            with self.dir_lock:
                try:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                except FileExistsError:
                    # Directory already exists, this is fine
                    pass
                except Exception as e:
                    print(f"Warning: Failed to create output directory {self.output_dir}: {e}")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Warning: Failed to save results to {output_file}: {e}")
                # Continue without failing
        
        # Calculate summary statistics
        total_tasks = len(results)
        total_tests = sum(len(r['test_results']) for r in results)
        total_passed = sum(sum(int(tr) for tr in r['test_results']) for r in results)
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Use lock for thread-safe console output
        with self.file_lock:
            print(f"  {code_technique} vs {test_technique}: {total_passed}/{total_tests} ({overall_pass_rate:.1%})")
        
        return {
            'code_technique': code_technique,
            'test_technique': test_technique,
            'total_tasks': total_tasks,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'overall_pass_rate': overall_pass_rate
        }
    
    def run_all_combinations(self) -> List[Dict]:
        """Run all combinations using thread pool."""
        print("Starting fixed cross-technique testing...")
        print(f"Using {self.max_workers} worker threads")
        print(f"Python executable: {self.test_env_python}")
        
        code_files = self.get_code_files()
        test_files = self.get_test_files()
        
        print(f"Found {len(code_files)} code files and {len(test_files)} test files")
        
        # Create all combinations
        combinations = [(code_file, test_file) for code_file in code_files for test_file in test_files]
        
        all_summaries = []
        
        # Filter out adversarial combinations
        valid_combinations = []
        for code_file, test_file in combinations:
            test_technique = self.extract_technique_name(test_file.name)
            if "adversarial" not in test_technique:
                valid_combinations.append((code_file, test_file))
        
        # Process combinations using thread pool for faster execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_combination = {}
            for code_file, test_file in valid_combinations:
                future = executor.submit(self.test_technique_pair_worker, (code_file, test_file))
                future_to_combination[future] = (code_file, test_file)
            
            # Process results as they complete
            with tqdm(total=len(valid_combinations), desc="Processing combinations") as pbar:
                for future in as_completed(future_to_combination):
                    code_file, test_file = future_to_combination[future]
                    code_technique = self.extract_technique_name(code_file.name)
                    test_technique = self.extract_technique_name(test_file.name)
                    
                    pbar.set_postfix_str(f"{code_technique} vs {test_technique}")
                    
                    try:
                        # Get results from completed future
                        results = future.result()
                        
                        # Save results (this needs to be thread-safe)
                        summary = self.save_results(results, code_technique, test_technique)
                        all_summaries.append(summary)
                        
                    except Exception as e:
                        print(f"\nError processing {code_technique} vs {test_technique}: {e}")
                        traceback.print_exc()
                    
                    # Force garbage collection
                    gc.collect()
                    pbar.update(1)
        
        return all_summaries
    
    def generate_final_report(self, summaries: List[Dict]):
        """Generate final report."""
        print(f"\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}")
        
        total_combinations = len(summaries)
        total_tests = sum(s['total_tests'] for s in summaries)
        total_passed = sum(s['total_passed'] for s in summaries)
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        print(f"Total combinations: {total_combinations}")
        print(f"Total tests: {total_tests}")
        print(f"Total passed: {total_passed}")
        print(f"Overall pass rate: {overall_pass_rate:.1%}")
        
        # Save comprehensive summary with thread safety
        with self.file_lock:
            # Ensure output directory exists with directory lock
            with self.dir_lock:
                try:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                except FileExistsError:
                    # Directory already exists, this is fine
                    pass
                except Exception as e:
                    print(f"Warning: Failed to create output directory {self.output_dir}: {e}")
            
            summary_file = self.output_dir / "fixed_comprehensive_summary.json"
            
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'overall_statistics': {
                            'total_combinations': total_combinations,
                            'total_tests': total_tests,
                            'total_passed': total_passed,
                            'overall_pass_rate': overall_pass_rate
                        },
                        'combination_results': summaries
                    }, f, indent=2)
                
                print(f"Summary saved to: {summary_file}")
            except Exception as e:
                print(f"Warning: Failed to save summary: {e}")
                print("Summary data:")
                print(f"  Total combinations: {total_combinations}")
                print(f"  Total tests: {total_tests}")
                print(f"  Total passed: {total_passed}")
                print(f"  Overall pass rate: {overall_pass_rate:.1%}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Cross-technique testing")
    parser.add_argument("--code-dir", required=True, help="Directory containing code files")
    parser.add_argument("--test-dir", required=True, help="Directory containing test files")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--max-workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Validate directories
    if not Path(args.code_dir).exists():
        print(f"Error: Code directory {args.code_dir} does not exist!")
        return 1
    
    if not Path(args.test_dir).exists():
        print(f"Error: Test directory {args.test_dir} does not exist!")
        return 1
    
    # Initialize and run tester
    tester = FixedCrossTechniqueTester(args.code_dir, args.test_dir, args.output_dir, args.max_workers)
    
    try:
        start_time = time.time()
        summaries = tester.run_all_combinations()
        tester.generate_final_report(summaries)
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal time: {elapsed_time/60:.1f} minutes")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())