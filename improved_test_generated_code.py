#!/usr/bin/env python3
"""
Improved test runner for BigCodeBench generated code against original test cases.
Runs each test in an isolated temporary environment with better error handling and debugging.
"""

import json
import os
import sys
import tempfile
import shutil
import subprocess
import venv
import ast
import traceback
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with: pip install tqdm")
    print("Falling back to basic progress reporting...")


class ImprovedIsolatedTestRunner:
    """Improved test runner for generated code in isolated environments with multithreading support."""
    
    def __init__(self, dataset_path: str, generated_code_dir: str, max_workers: int = None):
        self.dataset_path = Path(dataset_path)
        self.generated_code_dir = Path(generated_code_dir)
        self.results = {}
        self.shared_env_dir = None
        self.python_exe = None
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)  # Limit to 8 for safety
        self.print_lock = Lock()  # For thread-safe printing
        self.setup_lock = Lock()  # For thread-safe environment setup
        
    def load_dataset(self) -> Dict[str, Dict]:
        """Load the original dataset with test cases."""
        dataset = {}
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line.strip())
                    dataset[task['task_id']] = task
        return dataset
    
    def load_generated_code(self, file_path: Path) -> List[Dict]:
        """Load generated code from JSONL file."""
        generated_code = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        generated_code.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON on line {line_num} in {file_path}: {e}")
                        continue
        return generated_code
    
    def validate_code_syntax(self, code: str, task_id: str) -> Tuple[bool, str]:
        """Validate Python syntax of generated code."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
    
    def setup_shared_environment(self) -> bool:
        """Set up virtual environment in test_environment/venv with all dependencies."""
        # Use lock for thread-safety
        with self.setup_lock:
            if self.python_exe is not None and self.python_exe.exists():
                return True  # Already set up
                
            with self.print_lock:
                print("Setting up shared test environment...")
            
            # Setup virtual environment path
            self.shared_env_dir = Path("test_environment").absolute()
            
            # Determine python executable path in venv
            if sys.platform == "win32":
                self.python_exe = self.shared_env_dir / "Scripts" / "python.exe"
                pip_exe = self.shared_env_dir / "Scripts" / "pip.exe"
            else:
                self.python_exe = self.shared_env_dir / "bin" / "python"
                pip_exe = self.shared_env_dir / "bin" / "pip"
            
            try:
                # Check if virtual environment already exists and is valid
                if self.python_exe.exists():
                    # Test if key dependencies are available
                    result = subprocess.run([str(self.python_exe), "-c", "import psutil, numpy, pandas, matplotlib"], 
                                          capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        with self.print_lock:
                            print(f"  Reusing existing virtual environment: {self.shared_env_dir}")
                        return True
                    else:
                        with self.print_lock:
                            print(f"  Virtual environment exists but dependencies missing, recreating...")
                        shutil.rmtree(self.shared_env_dir)
                
                # Create new virtual environment  
                with self.print_lock:
                    print(f"  Creating virtual environment at: {self.shared_env_dir}")
                self.shared_env_dir.mkdir(exist_ok=True)
                venv.create(self.shared_env_dir, with_pip=True)
                
                # Verify python executable exists
                if not self.python_exe.exists():
                    with self.print_lock:
                        print(f"Error: Python executable not found at {self.python_exe}")
                    return False
                
                with self.print_lock:
                    print("  Upgrading pip...")
                subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], 
                             check=True, capture_output=True, text=True, timeout=360)
                
                # Install common dependencies
                with self.print_lock:
                    print("  Installing dependencies...")
                dependencies = [
                    "psutil", "numpy", "pandas", "matplotlib", "scikit-learn",
                    "requests", "beautifulsoup4", "lxml", "openpyxl", "seaborn",
                    "flask", "werkzeug", "pillow", "opencv-python", "wordcloud"
                ]
                
                subprocess.run([str(pip_exe), "install"] + dependencies, 
                             check=True, capture_output=True, text=True, timeout=300)
                
                # Verify key dependencies are installed
                result = subprocess.run([str(self.python_exe), "-c", 
                                       "import psutil, numpy, pandas, matplotlib, sklearn; print('Dependencies verified')"], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    with self.print_lock:
                        print(f"  Virtual environment ready: {self.shared_env_dir}")
                    return True
                else:
                    with self.print_lock:
                        print(f"  Error: Dependency verification failed")
                        print(f"  stderr: {result.stderr}")
                    return False
                    
            except Exception as e:
                with self.print_lock:
                    print(f"Failed to create virtual environment: {e}")
                return False
    
    def prepare_test_file(self, temp_dir: Path, task_data: Dict, generated_code: str) -> Tuple[Path, bool, str]:
        """Prepare a complete test file with generated code and test cases."""
        test_file = temp_dir / "test_task.py"
        
        try:
            # Validate generated code syntax first
            valid, error = self.validate_code_syntax(generated_code, task_data['task_id'])
            if not valid:
                return test_file, False, f"Generated code syntax error: {error}"
            
            # Extract imports from generated code
            imports = []
            code_lines = generated_code.strip().split('\n')
            for line in code_lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    imports.append(line)
            
            # Prepare the complete test file content
            test_content = []
            
            # Add imports
            test_content.extend(imports)
            test_content.append('')
            
            # Add generated code
            test_content.append(generated_code.strip())
            test_content.append('')
            
            # Add test cases
            test_code = task_data['test']
            test_content.append(test_code)
            test_content.append('')
            
            # Ensure unittest.main() is present
            if 'unittest.main()' not in test_code:
                test_content.append('if __name__ == "__main__":')
                test_content.append('    unittest.main()')
            
            # Write test file
            final_content = '\n'.join(test_content)
            
            # Validate final test file syntax
            valid, error = self.validate_code_syntax(final_content, task_data['task_id'])
            if not valid:
                return test_file, False, f"Combined test file syntax error: {error}"
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            return test_file, True, ""
            
        except Exception as e:
            return test_file, False, f"Error preparing test file: {str(e)}"
    
    def run_test_in_env(self, temp_dir: Path, test_file: Path) -> Tuple[bool, str, str]:
        """Run the test file in the shared environment."""
        original_cwd = None
        try:
            # Store original directory and change to temp directory for test execution
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            if self.python_exe is None:
                return False, "", "Python executable is None"
            
            # Run the test using the shared python environment
            result = subprocess.run(
                [str(self.python_exe), str(test_file)],
                capture_output=True,
                text=True,
                timeout=360,  # 6 minute timeout per test
                cwd=str(temp_dir)  # Explicitly set working directory
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Test timed out after 6 minutes"
        except Exception as e:
            return False, "", f"Error running test: {str(e)}"
        finally:
            # Always try to return to original directory, but handle if it was deleted
            if original_cwd is not None:
                try:
                    os.chdir(original_cwd)
                except (FileNotFoundError, OSError):
                    # If original directory was deleted, go to a safe directory
                    try:
                        os.chdir(Path.home())
                    except:
                        os.chdir("/tmp")  # Last resort
    
    def extract_detailed_error_info(self, stderr: str) -> Dict[str, str]:
        """Extract detailed error information from stderr."""
        error_info = {
            "error_type": "Unknown",
            "error_message": "",
            "traceback": "",
            "line_number": "",
            "specific_details": ""
        }
        
        lines = stderr.split('\n')
        
        # Find the main error line (usually the last non-empty line)
        main_error_line = ""
        for line in reversed(lines):
            if line.strip() and not line.startswith(' '):
                main_error_line = line.strip()
                break
        
        error_info["error_message"] = main_error_line
        
        # Extract traceback
        traceback_lines = []
        in_traceback = False
        for line in lines:
            if "Traceback (most recent call last):" in line:
                in_traceback = True
            if in_traceback:
                traceback_lines.append(line)
        
        if traceback_lines:
            error_info["traceback"] = '\n'.join(traceback_lines[-10:])  # Last 10 lines
        
        # Extract line number from traceback
        for line in lines:
            if "line " in line and "test_task.py" in line:
                try:
                    line_num = line.split("line ")[1].split(",")[0]
                    error_info["line_number"] = line_num
                except:
                    pass
        
        return error_info

    def categorize_failure(self, stderr: str, task_id: str) -> str:
        """Categorize test failure based on stderr output with detailed information."""
        if not stderr:
            return "Unknown test failure - no error output"
        
        # Extract detailed error information
        error_info = self.extract_detailed_error_info(stderr)
        stderr_lower = stderr.lower()
        
        # Check for specific error patterns with detailed explanations
        if "modulenotfounderror" in stderr_lower:
            if "no module named" in stderr_lower:
                try:
                    # Extract module name and provide installation suggestion
                    module_match = stderr.split("ModuleNotFoundError: No module named '")[1].split("'")[0]
                    return f"Missing module '{module_match}' - install with: pip install {module_match}"
                except:
                    return f"Module import error: {error_info['error_message']}"
            return f"Module import error: {error_info['error_message']}"
        
        elif "syntaxerror" in stderr_lower:
            # Extract detailed syntax error information
            syntax_details = ""
            for line in stderr.split('\n'):
                if "SyntaxError:" in line:
                    syntax_details = line.strip()
                    break
            line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
            return f"Syntax error{line_info}: {syntax_details}"
        
        elif "indentationerror" in stderr_lower:
            # Extract indentation error details
            indent_details = ""
            for line in stderr.split('\n'):
                if "IndentationError:" in line:
                    indent_details = line.strip()
                    break
            line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
            return f"Indentation error{line_info}: {indent_details}"
        
        elif "nameerror" in stderr_lower:
            # Extract variable/function name that's not defined
            name_details = ""
            for line in stderr.split('\n'):
                if "NameError:" in line:
                    name_details = line.strip()
                    break
            line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
            return f"Name error{line_info}: {name_details}"
        
        elif "attributeerror" in stderr_lower:
            # Extract detailed attribute error information
            attr_details = ""
            for line in stderr.split('\n'):
                if "AttributeError:" in line:
                    attr_details = line.strip()
                    break
            line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
            return f"Attribute error{line_info}: {attr_details}"
        
        elif "typeerror" in stderr_lower:
            # Extract detailed type error information
            type_details = ""
            for line in stderr.split('\n'):
                if "TypeError:" in line:
                    type_details = line.strip()
                    break
            line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
            return f"Type error{line_info}: {type_details}"
        
        elif "valueerror" in stderr_lower:
            # Extract value error details
            value_details = ""
            for line in stderr.split('\n'):
                if "ValueError:" in line:
                    value_details = line.strip()
                    break
            line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
            return f"Value error{line_info}: {value_details}"
        
        elif "keyerror" in stderr_lower:
            # Extract key error details
            key_details = ""
            for line in stderr.split('\n'):
                if "KeyError:" in line:
                    key_details = line.strip()
                    break
            line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
            return f"Key error{line_info}: {key_details}"
        
        elif "indexerror" in stderr_lower:
            # Extract index error details
            index_details = ""
            for line in stderr.split('\n'):
                if "IndexError:" in line:
                    index_details = line.strip()
                    break
            line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
            return f"Index error{line_info}: {index_details}"
        
        elif "filenotfounderror" in stderr_lower:
            # Extract file not found details
            file_details = ""
            for line in stderr.split('\n'):
                if "FileNotFoundError:" in line:
                    file_details = line.strip()
                    break
            return f"File not found: {file_details}"
        
        elif "permissionerror" in stderr_lower:
            # Extract permission error details
            perm_details = ""
            for line in stderr.split('\n'):
                if "PermissionError:" in line:
                    perm_details = line.strip()
                    break
            return f"Permission error: {perm_details}"
        
        elif "timeouterror" in stderr_lower or "timed out" in stderr_lower:
            return "Test timeout - execution exceeded 60 seconds"
        
        elif "fail:" in stderr_lower:
            # Extract detailed test failure information
            fail_details = []
            assertion_details = []
            
            lines = stderr.split('\n')
            for i, line in enumerate(lines):
                if 'FAIL:' in line:
                    fail_details.append(line.split('FAIL:')[1].strip())
                    # Look for assertion details in following lines
                    for j in range(i+1, min(i+10, len(lines))):
                        if lines[j].strip().startswith('AssertionError:'):
                            assertion_details.append(lines[j].strip())
                        elif lines[j].strip().startswith('assert'):
                            assertion_details.append(f"Failed assertion: {lines[j].strip()}")
            
            if fail_details:
                failure_summary = fail_details[0]
                if assertion_details:
                    return f"Test failure in {failure_summary}: {assertion_details[0]}"
                else:
                    return f"Test failure in {failure_summary}"
            return "Test assertion failed - no specific details found"
        
        elif "error:" in stderr_lower:
            # Extract detailed test error information
            error_details = []
            
            lines = stderr.split('\n')
            for i, line in enumerate(lines):
                if 'ERROR:' in line:
                    error_details.append(line.split('ERROR:')[1].strip())
                    # Look for the actual error in following lines
                    for j in range(i+1, min(i+5, len(lines))):
                        if any(err_type in lines[j] for err_type in ['Error:', 'Exception:']):
                            error_details.append(lines[j].strip())
                            break
            
            if error_details:
                test_name = error_details[0]
                error_desc = error_details[1] if len(error_details) > 1 else "Unknown error"
                return f"Test error in {test_name}: {error_desc}"
            return "Test execution error - no specific details found"
        
        else:
            # For any other errors, provide more context
            # Extract the most relevant error line
            relevant_lines = []
            for line in stderr.split('\n'):
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                    relevant_lines.append(line.strip())
            
            if relevant_lines:
                # Take the last meaningful error line
                main_error = relevant_lines[-1][:300]  # Limit length
                line_info = f" at line {error_info['line_number']}" if error_info['line_number'] else ""
                return f"Runtime error{line_info}: {main_error}"
            else:
                # Fallback to first few lines of stderr
                error_summary = stderr.strip()[:300].replace('\n', ' | ')
                return f"Unknown error: {error_summary}"
    
    def test_single_task(self, task_id: str, task_data: Dict, generated_code: str) -> Dict:
        """Test a single generated code against its original test case."""
        with self.print_lock:
            print(f"Testing {task_id}...")
        
        result = {
            'id': task_id,
            'test_result': 0,
            'failure_reason': None,
            'debug_info': {}  # Added for detailed debugging
        }
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            try:
                # Ensure shared environment is set up
                if not self.setup_shared_environment():
                    result['failure_reason'] = "Failed to create shared environment"
                    return result
                
                # Prepare test file
                test_file, file_valid, file_error = self.prepare_test_file(temp_dir, task_data, generated_code)
                
                if not file_valid:
                    result['failure_reason'] = file_error
                    result['debug_info'] = {
                        'stage': 'file_preparation',
                        'generated_code_preview': generated_code[:200] + "..." if len(generated_code) > 200 else generated_code,
                        'file_error': file_error
                    }
                    with self.print_lock:
                        print(f"  ❌ FAILED: {file_error}")
                    return result
                
                # Run test
                success, stdout, stderr = self.run_test_in_env(temp_dir, test_file)
                
                if success:
                    result['test_result'] = 1
                    result['failure_reason'] = None
                    with self.print_lock:
                        print(f"  ✅ PASSED")
                else:
                    result['test_result'] = 0
                    result['failure_reason'] = self.categorize_failure(stderr, task_id)
                    
                    # Add comprehensive debug information
                    result['debug_info'] = {
                        'stage': 'test_execution',
                        'stdout': stdout[:500] + "..." if len(stdout) > 500 else stdout,
                        'stderr': stderr[:1000] + "..." if len(stderr) > 1000 else stderr,
                        'generated_code_preview': generated_code[:300] + "..." if len(generated_code) > 300 else generated_code,
                        'test_code_preview': task_data['test'][:300] + "..." if len(task_data['test']) > 300 else task_data['test'],
                        'error_analysis': self.extract_detailed_error_info(stderr)
                    }
                    
                    with self.print_lock:
                        print(f"  ❌ FAILED: {result['failure_reason']}")
                
            except Exception as e:
                result['test_result'] = 0
                result['failure_reason'] = f"Framework error: {str(e)}"
                result['debug_info'] = {
                    'stage': 'framework_error',
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'traceback': traceback.format_exc()
                }
                with self.print_lock:
                    print(f"  💥 ERROR: {str(e)}")
        
        return result
    
    def test_single_task_wrapper(self, args):
        """Wrapper function for multiprocessing/threading."""
        task_id, task_data, generated_code = args
        return self.test_single_task(task_id, task_data, generated_code)
    
    def test_generated_file(self, file_path: Path) -> List[Dict]:
        """Test all generated code in a file against original test cases using multithreading."""
        print(f"\n{'='*60}")
        print(f"Testing file: {file_path.name}")
        print(f"Using {self.max_workers} worker threads")
        print(f"{'='*60}")
        
        # Load dataset and generated code
        dataset = self.load_dataset()
        generated_codes = self.load_generated_code(file_path)
        
        print(f"Loaded {len(dataset)} tasks from dataset")
        print(f"Loaded {len(generated_codes)} generated solutions")
        
        # Prepare tasks for parallel execution
        tasks = []
        matched_count = 0
        
        for gen_code in generated_codes:
            task_id = gen_code['task_id']
            
            if task_id not in dataset:
                print(f"Warning: Task {task_id} not found in dataset")
                continue
            
            matched_count += 1
            task_data = dataset[task_id]
            
            # Get the generated code
            if 'response_code' in gen_code and gen_code['response_code']:
                code = gen_code['response_code']
                tasks.append((task_id, task_data, code))
            else:
                print(f"Warning: No response_code found for task {task_id}")
                # Add failed result directly
                tasks.append((task_id, task_data, None))  # None indicates no code
        
        print(f"Prepared {len(tasks)} tasks for parallel execution")
        print(f"Starting parallel testing...")
        
        # Execute tests in parallel
        results = []
        completed_count = 0
        start_time = time.time()
        
        # Initialize progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=len(tasks),
                desc="Testing",
                unit="tests",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                task_id, task_data, code = task
                if code is None:
                    # Handle missing code case
                    result = {
                        'id': task_id,
                        'test_result': 0,
                        'failure_reason': 'No generated code found',
                        'debug_info': {'stage': 'no_code'}
                    }
                    results.append(result)
                    completed_count += 1
                    if TQDM_AVAILABLE:
                        pbar.update(1)
                        pbar.set_postfix_str("❌ No code")
                else:
                    future = executor.submit(self.test_single_task, task_id, task_data, code)
                    future_to_task[future] = task_id
            
            # Collect results as they complete
            passed_count = 0
            failed_count = 0
            
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    if result['test_result'] == 1:
                        passed_count += 1
                        status_icon = "✅"
                        status_text = "PASS"
                    else:
                        failed_count += 1
                        status_icon = "❌"
                        # Get short error type for display
                        failure_reason = result.get('failure_reason', 'Unknown')
                        if 'Syntax error' in failure_reason:
                            status_text = "SYNTAX"
                        elif 'Attribute error' in failure_reason:
                            status_text = "ATTR"
                        elif 'Type error' in failure_reason:
                            status_text = "TYPE"
                        elif 'Missing module' in failure_reason:
                            status_text = "MODULE"
                        elif 'Test failure' in failure_reason:
                            status_text = "ASSERT"
                        elif 'Test error' in failure_reason:
                            status_text = "ERROR"
                        else:
                            status_text = "FAIL"
                    
                    if TQDM_AVAILABLE:
                        pbar.update(1)
                        # Show current status and running statistics
                        pass_rate = (passed_count / completed_count * 100) if completed_count > 0 else 0
                        pbar.set_postfix_str(f"{status_icon} {status_text} | Pass: {passed_count}/{completed_count} ({pass_rate:.1f}%)")
                    else:
                        # Fallback progress update
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        eta = (len(tasks) - completed_count) / rate if rate > 0 else 0
                        
                        with self.print_lock:
                            print(f"Progress: {completed_count}/{len(tasks)} "
                                  f"({completed_count/len(tasks)*100:.1f}%) "
                                  f"Rate: {rate:.1f} tests/sec "
                                  f"ETA: {eta:.0f}s "
                                  f"Pass: {passed_count}/{completed_count}")
                        
                except Exception as e:
                    if not TQDM_AVAILABLE:
                        with self.print_lock:
                            print(f"Error processing task {task_id}: {e}")
                    
                    # Add error result
                    result = {
                        'id': task_id,
                        'test_result': 0,
                        'failure_reason': f'Execution error: {str(e)}',
                        'debug_info': {
                            'stage': 'parallel_execution_error',
                            'exception': str(e),
                            'traceback': traceback.format_exc()
                        }
                    }
                    results.append(result)
                    completed_count += 1
                    failed_count += 1
                    
                    if TQDM_AVAILABLE:
                        pbar.update(1)
                        pbar.set_postfix_str(f"💥 ERROR | Pass: {passed_count}/{completed_count}")
        
        # Close progress bar
        if TQDM_AVAILABLE:
            pbar.close()
        
        # Sort results by task ID to maintain order
        results.sort(key=lambda x: x['id'])
        
        elapsed_total = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Testing completed for {file_path.name}")
        print(f"Matched {matched_count} tasks")
        print(f"Total time: {elapsed_total:.1f}s")
        print(f"Average rate: {len(tasks)/elapsed_total:.1f} tests/sec")
        print(f"{'='*60}")
        
        return results
    
    def cleanup_shared_environment(self):
        """Clean up the shared environment."""
        if hasattr(self, 'shared_env_dir') and self.shared_env_dir and self.shared_env_dir.exists():
            shutil.rmtree(self.shared_env_dir)
            print(f"Virtual environment cleaned up: {self.shared_env_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Test Generated Code with Multithreading")
    parser.add_argument("--dataset", required=True,
                       help="Path to BigCodeBench dataset JSONL file")
    parser.add_argument("--generated", required=True,
                       help="Directory containing generated code JSONL files")
    parser.add_argument("--file", 
                       help="Test only a specific file")
    parser.add_argument("--output-dir", default="test_result",
                       help="Directory to save test results")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up test environment after completion")
    parser.add_argument("--workers", type=int, default=None,
                       help=f"Number of worker threads (default: min(CPU_count, 8) = {min(multiprocessing.cpu_count(), 8)})")
    parser.add_argument("--sequential", action="store_true",
                       help="Run tests sequentially instead of in parallel (for debugging)")
    
    args = parser.parse_args()
    
    # Set max_workers based on args
    if args.sequential:
        max_workers = 1
        print("Running in sequential mode for debugging")
    else:
        max_workers = args.workers
        if max_workers:
            print(f"Using {max_workers} worker threads")
        else:
            max_workers = min(multiprocessing.cpu_count(), 8)
            print(f"Auto-detected {max_workers} worker threads")
    
    # Initialize test runner
    runner = ImprovedIsolatedTestRunner(args.dataset, args.generated, max_workers)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if args.file:
            # Test single file
            file_path = Path(args.generated) / args.file
            if not file_path.exists():
                print(f"Error: File {file_path} does not exist")
                sys.exit(1)
            
            results = runner.test_generated_file(file_path)
            
            # Save results
            output_file = output_dir / file_path.name
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            # Save detailed debug information for failed tests
            debug_file = output_dir / f"debug_{file_path.name}"
            failed_results = [r for r in results if r['test_result'] == 0 and 'debug_info' in r]
            if failed_results:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_results, f, indent=2)
                print(f"Debug information saved to: {debug_file}")
            
            # Print summary
            total = len(results)
            passed = sum(1 for r in results if r['test_result'] == 1)
            print(f"\nSummary for {file_path.name}:")
            print(f"Total tests: {total}")
            print(f"Passed: {passed}")
            print(f"Pass rate: {passed/total*100:.1f}%")
            print(f"Results saved to: {output_file}")
            
            # Print error breakdown
            error_types = {}
            for result in results:
                if result['test_result'] == 0:
                    failure_reason = result.get('failure_reason', 'Unknown')
                    # Categorize by error type
                    if 'Syntax error' in failure_reason:
                        error_type = 'Syntax Error'
                    elif 'Attribute error' in failure_reason:
                        error_type = 'Attribute Error'
                    elif 'Type error' in failure_reason:
                        error_type = 'Type Error'
                    elif 'Name error' in failure_reason:
                        error_type = 'Name Error'
                    elif 'Missing module' in failure_reason:
                        error_type = 'Missing Module'
                    elif 'Test failure' in failure_reason:
                        error_type = 'Test Assertion Failure'
                    elif 'Test error' in failure_reason:
                        error_type = 'Test Execution Error'
                    else:
                        error_type = 'Other'
                    
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error_types:
                print(f"\nError breakdown:")
                for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {error_type}: {count}")
            
            # Show a few example failures with details
            failed_with_details = [r for r in results if r['test_result'] == 0][:3]
            if failed_with_details:
                print(f"\nExample failures (first 3):")
                for i, result in enumerate(failed_with_details, 1):
                    print(f"  {i}. {result['id']}: {result['failure_reason']}")
            
            if failed_results:
                print(f"\nFor detailed debugging information, see: {debug_file}")
            
        else:
            # Test all files
            generated_dir = Path(args.generated)
            code_files = list(generated_dir.glob("code_*.jsonl"))
            
            if not code_files:
                print(f"No code_*.jsonl files found in {generated_dir}")
                sys.exit(1)
            
            all_results = []
            
            for code_file in sorted(code_files):
                results = runner.test_generated_file(code_file)
                all_results.extend(results)
                
                # Save individual file results
                output_file = output_dir / code_file.name
                with open(output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
                
                # Print file summary
                total = len(results)
                passed = sum(1 for r in results if r['test_result'] == 1)
                print(f"\nSummary for {code_file.name}:")
                print(f"Total tests: {total}")
                print(f"Passed: {passed}")
                print(f"Pass rate: {passed/total*100:.1f}%")
            
            # Print overall summary
            total_all = len(all_results)
            passed_all = sum(1 for r in all_results if r['test_result'] == 1)
            print(f"\n{'='*60}")
            print("OVERALL SUMMARY")
            print(f"{'='*60}")
            print(f"Total tests: {total_all}")
            print(f"Passed: {passed_all}")
            print(f"Pass rate: {passed_all/total_all*100:.1f}%")
            print(f"Results saved to: {output_dir}")
    
    finally:
        # Cleanup if requested
        if args.cleanup:
            runner.cleanup_shared_environment()


if __name__ == "__main__":
    main()