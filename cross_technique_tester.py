#!/usr/bin/env python3
"""
Cross-technique testing framework for BigCodeBench.
Tests each generated code technique against each generated test technique.
Creates 64 combinations (8 code × 8 test techniques) and evaluates them in isolation.
"""

import json
import os
import sys
import tempfile
import shutil
import subprocess
import venv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback
import re
import resource
import psutil
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from tqdm import tqdm


class CrossTechniqueTester:
    """Tests generated code from one technique against tests from another technique."""
    
    def __init__(self, code_dir: str, test_dir: str, output_dir: str = "cross_technique_results", max_workers: int = 2):
        self.code_dir = Path(code_dir)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Resource management
        self.max_workers = max_workers
        self.max_memory_mb = 2048  # 2GB memory limit per process
        self.system_memory_threshold = 0.9  # Pause if system memory > 90%
        
        # Create shared virtual environment
        self.shared_env_dir = None
        self.python_exe = None
        self.setup_shared_environment()
    
    def setup_shared_environment(self) -> bool:
        """Create a shared virtual environment with all required packages."""
        print("Setting up shared test environment...")
        
        # Create environment in a persistent location
        self.shared_env_dir = Path("cross_test_environment").absolute()
        
        try:
            # Reuse existing environment if it exists and is valid
            if self.shared_env_dir.exists():
                python_exe = self.shared_env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
                if python_exe.exists():
                    print("  Reusing existing virtual environment...")
                    self.python_exe = python_exe
                    return True
                else:
                    print("  Existing environment invalid, recreating...")
                    shutil.rmtree(self.shared_env_dir)
            
            # Create virtual environment only if needed
            print("  Creating new virtual environment...")
            venv.create(self.shared_env_dir, with_pip=True)
            
            # Get paths to the virtual environment
            if sys.platform == "win32":
                self.python_exe = self.shared_env_dir / "Scripts" / "python.exe"
                pip_exe = self.shared_env_dir / "Scripts" / "pip.exe"
            else:
                self.python_exe = self.shared_env_dir / "bin" / "python"
                pip_exe = self.shared_env_dir / "bin" / "pip"
            
            print("  Upgrading pip...")
            subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], 
                         check=True, capture_output=True, text=True)
            
            # Install requirements from requirements-eval.txt
            requirements_file = Path("Requirements/requirements-eval.txt")
            if requirements_file.exists():
                print(f"  Installing dependencies from {requirements_file}")
                subprocess.run([str(pip_exe), "install", "-r", str(requirements_file)], 
                             check=True, capture_output=True, text=True)
                print("  Dependencies installed successfully!")
            else:
                print(f"Warning: Requirements file {requirements_file} not found")
            
            return True
            
        except Exception as e:
            print(f"Failed to create shared environment: {e}")
            return False
    
    def cleanup_shared_environment(self):
        """Clean up the shared environment."""
        if self.shared_env_dir and self.shared_env_dir.exists():
            shutil.rmtree(self.shared_env_dir)
            print("Shared environment cleaned up.")
    
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
    
    def get_code_files(self) -> List[Path]:
        """Get all code_*.jsonl files."""
        return sorted(list(self.code_dir.glob("code_*.jsonl")))
    
    def get_test_files(self) -> List[Path]:
        """Get all test_*.jsonl files."""
        return sorted(list(self.test_dir.glob("test_*.jsonl")))
    
    def prepare_combined_test_file(self, temp_dir: Path, code_entry: Dict, test_entry: Dict) -> Path:
        """Prepare a complete test file combining generated code and generated tests."""
        test_file = temp_dir / "combined_test.py"
        
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
                if 'your_module' not in line and 'solution' not in line:
                    imports.add(line)
        
        # Standardize function names to ensure compatibility
        # Code uses task_func, tests use test_func - create an alias
        processed_code = generated_code.strip()
        processed_test = generated_test.strip()
        
        # Prepare the complete test file content
        test_content = []
        
        # Add all unique imports
        test_content.extend(sorted(imports))
        test_content.append('')
        
        # Add generated code (this defines task_func)
        test_content.append('# Generated code:')
        test_content.append(processed_code)
        test_content.append('')
        
        # Create alias from task_func to test_func for compatibility
        test_content.append('# Function alias for test compatibility:')
        test_content.append('test_func = task_func')
        test_content.append('')
        
        # Add generated test (this should call test_func)
        test_content.append('# Generated tests:')
        test_content.append(processed_test)
        test_content.append('')
        
        # Ensure unittest.main() is present
        if 'unittest.main()' not in generated_test:
            test_content.append('if __name__ == "__main__":')
            test_content.append('    unittest.main()')
        
        # Write test file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_content))
        
        return test_file
    
    def check_system_resources(self) -> bool:
        """Check if system has enough resources to continue."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.system_memory_threshold:
                print(f"  ⚠️  System memory usage high ({memory_percent:.1%}), pausing...")
                time.sleep(0.5)  # Reduced sleep to prevent system sleep
                return False
            return True
        except Exception:
            return True  # Continue if we can't check
    
    def set_resource_limits(self):
        """Set resource limits for the current process."""
        try:
            # Set memory limit (soft limit)
            memory_limit = self.max_memory_mb * 1024 * 1024  # Convert to bytes
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        except Exception as e:
            print(f"  Warning: Could not set resource limits: {e}")
    
    def run_test_in_env(self, temp_dir: Path, test_file: Path) -> Tuple[bool, List[str], str]:
        """Run the test file and return results."""
        try:
            # Set resource limits
            self.set_resource_limits()
            
            # Change to temp directory for test execution
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            # Run the test using the shared python environment with reduced timeout
            result = subprocess.run(
                [str(self.python_exe), str(test_file), '-v'],
                capture_output=True,
                text=True,
                timeout=30  # Reduced timeout to 30 seconds
            )
            
            os.chdir(original_cwd)
            
            # Parse test results
            test_results, failure_reason = self.parse_test_output(result.stdout, result.stderr, result.returncode)
            success = result.returncode == 0
            
            return success, test_results, failure_reason
            
        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            return False, [], "Test timed out after 30 seconds"
        except Exception as e:
            os.chdir(original_cwd)
            return False, [], f"Error running test: {str(e)}"
    
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
        print(f"  Testing {task_id} (code: {code_technique}, test: {test_technique})")
        
        result = {
            'task_id': task_id,
            'code_technique': code_technique,
            'test_technique': test_technique,
            'test_results': [],
            'pass_rate': 0.0,
            'failure_reason': None
        }
        
        # Check system resources before proceeding
        if not self.check_system_resources():
            time.sleep(0.5)  # Brief pause if resources are low
        
        # Create temporary directory for this specific test
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
                
            except Exception as e:
                result['test_results'] = ["0"]
                result['pass_rate'] = 0.0
                result['failure_reason'] = f"Framework error: {str(e)}"
                print(f"    ❌ ERROR: {str(e)}")
            finally:
                # Force garbage collection to free memory
                gc.collect()
        
        return result
    
    def test_technique_pair(self, code_file: Path, test_file: Path) -> List[Dict]:
        """Test all tasks from one code technique against one test technique."""
        code_technique = self.extract_technique_name(code_file.name)
        test_technique = self.extract_technique_name(test_file.name)
        
        # Load data
        code_data = self.load_jsonl_file(code_file)
        test_data = self.load_jsonl_file(test_file)
        
        results = []
        
        # Find common task IDs
        common_tasks = set(code_data.keys()) & set(test_data.keys())
        
        # Use tqdm for task progress within each combination
        with tqdm(sorted(common_tasks), desc=f"  Tasks ({code_technique} vs {test_technique})", 
                 leave=False, unit="task") as task_pbar:
            for task_id in task_pbar:
                task_pbar.set_description(f"  Testing {task_id[:20]}...")
                
                code_entry = code_data[task_id]
                test_entry = test_data[task_id]
                
                result = self.test_code_against_test(code_entry, test_entry, code_technique, test_technique)
                results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], code_technique: str, test_technique: str):
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
        
        print(f"  Results saved to: {output_file}")
        print(f"  Summary: {total_passed}/{total_tests} tests passed ({overall_pass_rate:.1%})")
        
        return {
            'code_technique': code_technique,
            'test_technique': test_technique,
            'total_tasks': total_tasks,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'overall_pass_rate': overall_pass_rate
        }
    
    def run_all_combinations(self) -> Dict:
        """Run all combinations of code vs test techniques with resource management and progress bars."""
        print("Starting cross-technique testing...")
        print(f"Code directory: {self.code_dir}")
        print(f"Test directory: {self.test_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Max workers: {self.max_workers}")
        
        code_files = self.get_code_files()
        test_files = self.get_test_files()
        
        print(f"\nFound {len(code_files)} code files and {len(test_files)} test files")
        print(f"Code files: {[self.extract_technique_name(f.name) for f in code_files]}")
        print(f"Test files: {[self.extract_technique_name(f.name) for f in test_files]}")
        print(f"Will run {len(code_files)} × {len(test_files)} = {len(code_files) * len(test_files)} combinations")
        
        all_summaries = []
        total_combinations = len(code_files) * len(test_files)
        
        # Create combinations list
        combinations = [(i, code_file, j, test_file) for i, code_file in enumerate(code_files, 1) 
                       for j, test_file in enumerate(test_files, 1)]
        
        # Use tqdm for overall progress
        with tqdm(total=total_combinations, desc="Cross-technique testing", unit="combo") as pbar:
            for i, code_file, j, test_file in combinations:
                code_technique = self.extract_technique_name(code_file.name)
                test_technique = self.extract_technique_name(test_file.name)
                
                combination_num = (i-1)*len(test_files) + j
                pbar.set_description(f"Testing {code_technique} vs {test_technique}")
                
                # Check system resources before each combination
                while not self.check_system_resources():
                    time.sleep(1)  # Brief wait for system resources
                
                # Test this combination
                results = self.test_technique_pair(code_file, test_file)
                
                # Save results
                summary = self.save_results(results, code_technique, test_technique)
                all_summaries.append(summary)
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'pass_rate': f"{summary['overall_pass_rate']:.1%}",
                    'tests': f"{summary['total_passed']}/{summary['total_tests']}"
                })
                
                # Force garbage collection periodically
                if combination_num % 10 == 0:
                    gc.collect()
        
        return all_summaries
    
    def generate_final_report(self, summaries: List[Dict]):
        """Generate a final comprehensive report."""
        print(f"\n{'='*80}")
        print("FINAL COMPREHENSIVE REPORT")
        print(f"{'='*80}")
        
        # Overall statistics
        total_combinations = len(summaries)
        total_tasks_across_all = sum(s['total_tasks'] for s in summaries)
        total_tests_across_all = sum(s['total_tests'] for s in summaries)
        total_passed_across_all = sum(s['total_passed'] for s in summaries)
        overall_pass_rate = total_passed_across_all / total_tests_across_all if total_tests_across_all > 0 else 0.0
        
        print(f"Total combinations tested: {total_combinations}")
        print(f"Total task instances: {total_tasks_across_all}")
        print(f"Total individual tests: {total_tests_across_all}")
        print(f"Total tests passed: {total_passed_across_all}")
        print(f"Overall pass rate: {overall_pass_rate:.1%}")
        
        # System resource usage summary
        memory_info = psutil.virtual_memory()
        print(f"\nSystem Resources:")
        print(f"Memory usage: {memory_info.percent:.1f}%")
        print(f"Available memory: {memory_info.available / (1024**3):.1f} GB")
        
        print(f"\nDetailed Results by Combination:")
        print(f"{'Code Technique':<25} {'Test Technique':<25} {'Pass Rate':<10} {'Tests':<10}")
        print("-" * 80)
        
        for summary in sorted(summaries, key=lambda x: x['overall_pass_rate'], reverse=True):
            print(f"{summary['code_technique']:<25} {summary['test_technique']:<25} "
                  f"{summary['overall_pass_rate']:>7.1%} {summary['total_passed']:>4}/{summary['total_tests']:<4}")
        
        # Save comprehensive summary
        summary_file = self.output_dir / "comprehensive_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'overall_statistics': {
                    'total_combinations': total_combinations,
                    'total_tasks': total_tasks_across_all,
                    'total_tests': total_tests_across_all,
                    'total_passed': total_passed_across_all,
                    'overall_pass_rate': overall_pass_rate
                },
                'combination_results': summaries
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nComprehensive summary saved to: {summary_file}")
        print(f"Individual results saved to: {self.output_dir}/")
        
        # Final cleanup
        gc.collect()


def main():
    """Main function to run cross-technique testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-technique testing for BigCodeBench")
    parser.add_argument("--code-dir", required=True,
                       help="Directory containing standardized code files")
    parser.add_argument("--test-dir", required=True,
                       help="Directory containing standardized test files")
    parser.add_argument("--output-dir", default="cross_technique_results",
                       help="Output directory for results")
    parser.add_argument("--max-workers", type=int, default=2,
                       help="Maximum number of worker processes (default: 2)")
    
    args = parser.parse_args()
    
    # Validate directories
    code_dir = Path(args.code_dir)
    test_dir = Path(args.test_dir)
    
    if not code_dir.exists():
        print(f"Error: Code directory {code_dir} does not exist!")
        sys.exit(1)
    
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist!")
        sys.exit(1)
    
    # Initialize and run cross-technique tester
    tester = CrossTechniqueTester(code_dir, test_dir, args.output_dir, args.max_workers)
    
    # Display system info
    print(f"System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"Max Workers: {args.max_workers}")
    print(f"Memory Threshold: {tester.system_memory_threshold:.0%}")
    print(f"Process Memory Limit: {tester.max_memory_mb} MB")
    
    try:
        start_time = time.time()
        
        # Run all combinations
        summaries = tester.run_all_combinations()
        
        # Generate final report
        tester.generate_final_report(summaries)
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time/60:.1f} minutes")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        # Clean up shared environment
        tester.cleanup_shared_environment()
        print("\nCleanup completed.")


if __name__ == "__main__":
    main()