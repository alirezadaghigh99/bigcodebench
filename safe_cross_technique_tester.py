#!/usr/bin/env python3
"""
Safe Cross-technique testing framework for BigCodeBench.
Uses threading instead of multiprocessing and proper isolation to prevent system crashes.
"""

import json
import os
import sys
import tempfile
import shutil
import subprocess
import threading
import time
import gc
import signal
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import resource


class SafeIsolatedRunner:
    """Runs code in a safe isolated environment with strict resource limits."""
    
    def __init__(self, timeout: int = 30, max_memory_mb: int = 2048, python_exe: str = None):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.python_exe = python_exe or sys.executable
        
    @contextmanager
    def isolated_environment(self):
        """Create a completely isolated temporary environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create isolated Python environment
            isolated_script = temp_path / "isolated_runner.py"
            self._create_isolated_runner_script(isolated_script)
            
            yield temp_path, isolated_script
    
    def _create_isolated_runner_script(self, script_path: Path):
        """Create a script that runs in complete isolation."""
        script_content = f'''
import sys
import os
import signal
import resource
import traceback
import gc
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Configure matplotlib to use non-interactive backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Prevent matplotlib from trying to create display
os.environ['DISPLAY'] = ''
os.environ['MPLBACKEND'] = 'Agg'

def set_limits():
    """Set strict resource limits."""
    try:
        # Memory limit (in bytes) - only if positive
        memory_limit = {self.max_memory_mb} * 1024 * 1024
        if {self.max_memory_mb} > 0:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        
        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, ({self.timeout}, {self.timeout}))
        
        # File descriptor limit (increased for complex tests)
        resource.setrlimit(resource.RLIMIT_NOFILE, (500, 500))
        
    except Exception as e:
        print(f"Warning: Could not set resource limits: {{e}}", file=sys.stderr)

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Test execution timed out")

def safe_exec_test(test_file_path):
    """Execute test file safely with all protections."""
    try:
        # Set resource limits
        set_limits()
        
        # Set timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm({self.timeout})
        
        # Redirect stdout/stderr to capture output
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Create isolated namespace
        test_namespace = {{
            '__name__': '__main__',
            '__file__': test_file_path,
        }}
        
        # Execute test file
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            with open(test_file_path, 'r') as f:
                test_code = f.read()
            
            # Execute the test code
            exec(test_code, test_namespace)
        
        # Cancel alarm
        signal.alarm(0)
        
        # Get outputs
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        # Return success
        return {{
            'success': True,
            'stdout': stdout_output,
            'stderr': stderr_output,
            'returncode': 0
        }}
        
    except TimeoutError:
        return {{
            'success': False,
            'stdout': '',
            'stderr': 'Test timed out',
            'returncode': 124
        }}
    except MemoryError:
        return {{
            'success': False,
            'stdout': '',
            'stderr': 'Memory limit exceeded',
            'returncode': 125
        }}
    except Exception as e:
        return {{
            'success': False,
            'stdout': '',
            'stderr': f'Error: {{str(e)}}\\n{{traceback.format_exc()}}',
            'returncode': 1
        }}
    finally:
        # Cleanup
        signal.alarm(0)
        plt.close('all')  # Close all matplotlib figures
        gc.collect()

if __name__ == '__main__':
    test_file = sys.argv[1]
    result = safe_exec_test(test_file)
    
    # Print result as JSON
    import json
    print(json.dumps(result))
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
    
    def run_test_safely(self, test_file_path: Path) -> Dict:
        """Run a test file in complete isolation."""
        try:
            with self.isolated_environment() as (temp_dir, isolated_script):
                # Copy test file to isolated environment
                isolated_test_file = temp_dir / "test_to_run.py"
                shutil.copy2(test_file_path, isolated_test_file)
                
                # Run isolated test
                result = subprocess.run(
                    [self.python_exe, str(isolated_script), str(isolated_test_file)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 5,  # Extra buffer for subprocess
                    cwd=str(temp_dir)
                )
                
                # Parse result
                if result.returncode == 0:
                    try:
                        test_result = json.loads(result.stdout.strip())
                        return test_result
                    except json.JSONDecodeError:
                        return {
                            'success': False,
                            'stdout': result.stdout,
                            'stderr': result.stderr,
                            'returncode': result.returncode
                        }
                else:
                    return {
                        'success': False,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode
                    }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Subprocess timeout',
                'returncode': 124
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Runner error: {str(e)}',
                'returncode': 1
            }


class SafeCrossTechniqueTester:
    """Thread-based safe cross-technique tester."""
    
    def __init__(self, code_dir: str, test_dir: str, output_dir: str = "cross_technique_results", max_workers: int = 2, max_memory_mb: int = 2048):
        self.code_dir = Path(code_dir)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup shared environment (reuse from original)
        self.shared_env_dir = Path("cross_test_environment").absolute()
        self.python_exe = self.shared_env_dir / "bin" / "python"
        
        # Threading configuration
        self.max_workers = max_workers
        self.runner = SafeIsolatedRunner(max_memory_mb=max_memory_mb, python_exe=str(self.python_exe))
        
        # Resource monitoring
        self.memory_threshold = 0.85  # 85% memory threshold
        self.check_interval = 1.0  # Check every second
        
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
    
    def prepare_combined_test_file(self, code_entry: Dict, test_entry: Dict) -> str:
        """Prepare a complete test file combining generated code and generated tests."""
        # Get the generated code
        generated_code = code_entry.get('response_code', '')
        if not generated_code:
            raise ValueError("No generated code found")
        
        # Get the generated test
        generated_test = test_entry.get('response_code', '')
        if not generated_test:
            raise ValueError("No generated test found")
        
        # Extract imports from both code and test
        imports = set()
        
        # Extract imports from generated code
        for line in generated_code.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.add(line)
        
        # Extract imports from generated test
        for line in generated_test.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Skip problematic imports that were removed
                if ('your_module' not in line and 'solution' not in line and 
                    'module' not in line and 'ftp_download' not in line and
                    'test_func' not in line and 'task_func' not in line):
                    imports.add(line)
        
        # Ensure matplotlib is properly configured
        imports.add('import matplotlib')
        imports.add('matplotlib.use("Agg")')
        
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
        
        # Add generated code (this defines task_func)
        test_content.append('# Generated code:')
        test_content.append(generated_code)
        test_content.append('')
        
        # Create alias from task_func to test_func for compatibility
        test_content.append('# Function alias for test compatibility:')
        test_content.append('try:')
        test_content.append('    test_func = task_func')
        test_content.append('except NameError:')
        test_content.append('    pass  # task_func not defined')
        test_content.append('')
        
        # Add generated test (this should call test_func)
        test_content.append('# Generated tests:')
        test_content.append(generated_test)
        test_content.append('')
        
        # Ensure unittest.main() is present
        if 'unittest.main()' not in generated_test:
            test_content.append('if __name__ == "__main__":')
            test_content.append('    unittest.main()')
        
        return '\n'.join(test_content)
    
    def check_system_resources(self) -> bool:
        """Check if system has enough resources."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent < self.memory_threshold
        except:
            return True
    
    def parse_test_output(self, result: Dict) -> Tuple[List[str], Optional[str]]:
        """Parse test result to extract individual test results."""
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        returncode = result.get('returncode', 1)
        
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
            
            # Extract failure reason
            if 'FAILED' in combined_output:
                failure_reason = "Test failures detected"
            elif 'ERROR' in combined_output:
                failure_reason = "Test errors detected"
            elif 'ModuleNotFoundError' in combined_output:
                failure_reason = "Missing module dependency"
            elif 'SyntaxError' in combined_output:
                failure_reason = "Syntax error in code"
            elif 'NameError' in combined_output:
                failure_reason = "Name error - function not defined correctly"
            else:
                failure_reason = "Unknown test failure"
        
        # Ensure we have at least one result
        if not test_results:
            test_results = ["0"]  # Default to failed if we can't parse
        
        return test_results, failure_reason
    
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
            'failure_reason': None
        }
        
        try:
            # Check system resources before proceeding
            if not self.check_system_resources():
                result['test_results'] = ["0"]
                result['failure_reason'] = "System resources insufficient"
                return result
            
            # Prepare combined test content
            test_content = self.prepare_combined_test_file(code_entry, test_entry)
            
            # Create temporary test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_content)
                test_file_path = Path(f.name)
            
            try:
                # Run test safely
                test_result = self.runner.run_test_safely(test_file_path)
                
                # Parse results
                test_results, failure_reason = self.parse_test_output(test_result)
                
                result['test_results'] = test_results
                result['pass_rate'] = sum(int(r) for r in test_results) / len(test_results) if test_results else 0.0
                result['failure_reason'] = failure_reason
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(test_file_path)
                except:
                    pass
        
        except Exception as e:
            result['test_results'] = ["0"]
            result['pass_rate'] = 0.0
            result['failure_reason'] = f"Framework error: {str(e)}"
        
        return result
    
    def test_technique_pair_worker(self, args) -> List[Dict]:
        """Worker function for testing a technique pair."""
        code_file, test_file = args
        code_technique = self.extract_technique_name(code_file.name)
        test_technique = self.extract_technique_name(test_file.name)
        
        print(f"Testing {code_technique} vs {test_technique}")
        
        # Load data
        code_data = self.load_jsonl_file(code_file)
        test_data = self.load_jsonl_file(test_file)
        
        results = []
        
        # Find common task IDs
        common_tasks = set(code_data.keys()) & set(test_data.keys())
        
        for task_id in sorted(common_tasks):
            if not self.check_system_resources():
                print(f"  Skipping {task_id} - insufficient resources")
                continue
                
            code_entry = code_data[task_id]
            test_entry = test_data[task_id]
            
            result = self.test_code_against_test(code_entry, test_entry, code_technique, test_technique)
            results.append(result)
            
            # Brief pause between tests
            time.sleep(0.1)
        
        return results
    
    def save_results(self, results: List[Dict], code_technique: str, test_technique: str) -> Dict:
        """Save results for a technique pair."""
        filename = f"code_{code_technique}_test_{test_technique}.jsonl"
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Calculate summary statistics
        total_tasks = len(results)
        total_tests = sum(len(r['test_results']) for r in results)
        total_passed = sum(sum(int(tr) for tr in r['test_results']) for r in results)
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
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
        print("Starting safe cross-technique testing...")
        print(f"Using {self.max_workers} worker threads")
        
        code_files = self.get_code_files()
        test_files = self.get_test_files()
        
        print(f"Found {len(code_files)} code files and {len(test_files)} test files")
        
        # Create all combinations
        combinations = [(code_file, test_file) for code_file in code_files for test_file in test_files]
        
        all_summaries = []
        
        # Use ThreadPoolExecutor for safer resource management
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_combo = {
                executor.submit(self.test_technique_pair_worker, combo): combo 
                for combo in combinations
            }
            
            # Process results as they complete
            for future in as_completed(future_to_combo):
                combo = future_to_combo[future]
                code_file, test_file = combo
                
                try:
                    results = future.result()
                    code_technique = self.extract_technique_name(code_file.name)
                    test_technique = self.extract_technique_name(test_file.name)
                    
                    # Save results
                    summary = self.save_results(results, code_technique, test_technique)
                    all_summaries.append(summary)
                    
                except Exception as e:
                    print(f"Error processing {combo}: {e}")
                
                # Force garbage collection
                gc.collect()
        
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
        
        # Save comprehensive summary
        summary_file = self.output_dir / "safe_comprehensive_summary.json"
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


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Cross-technique testing")
    parser.add_argument("--code-dir", default="generation_output/o4_standardized")
    parser.add_argument("--test-dir", default="generation_output/o4_test_standardized")
    parser.add_argument("--output-dir", default="cross_technique_results")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-memory-mb", type=int, default=2048, 
                       help="Maximum memory per test in MB (default: 2048, 0 to disable)")
    
    args = parser.parse_args()
    
    # Validate directories
    if not Path(args.code_dir).exists():
        print(f"Error: Code directory {args.code_dir} does not exist!")
        return 1
    
    if not Path(args.test_dir).exists():
        print(f"Error: Test directory {args.test_dir} does not exist!")
        return 1
    
    # Initialize and run tester
    tester = SafeCrossTechniqueTester(args.code_dir, args.test_dir, args.output_dir, args.max_workers, args.max_memory_mb)
    
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