#!/usr/bin/env python3
"""
Test runner for BigCodeBench generated code against original test cases.
Runs each test in an isolated temporary environment with proper dependency management.
"""

import json
import os
import sys
import tempfile
import shutil
import subprocess
import venv
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback


class IsolatedTestRunner:
    """Runs tests for generated code in isolated environments."""
    
    def __init__(self, dataset_path: str, generated_code_dir: str, output_dir: str = "test_result"):
        self.dataset_path = Path(dataset_path)
        self.generated_code_dir = Path(generated_code_dir)
        self.output_dir = Path(output_dir)
        self.results = {}
        self.shared_env_dir = None
        self.python_exe = None
        
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
                        print(f"Line content: {line[:100]}...")
                        continue
        return generated_code
    
    def setup_shared_environment(self) -> bool:
        """Set up virtual environment in test_environment/venv with all dependencies."""
        if self.python_exe is not None:
            return True  # Already set up
            
        print("Setting up shared test environment...")
        
        # Setup virtual environment path
        self.shared_env_dir = Path("test_environment").absolute()
        
        # Determine python executable path in venv
        if sys.platform == "win32":
            self.python_exe = self.shared_env_dir / "Scripts" / "python.exe"
            pip_exe = self.shared_env_dir / "Scripts" / "pip.exe"
            activate_script = self.shared_env_dir / "Scripts" / "activate"
        else:
            self.python_exe = self.shared_env_dir / "bin" / "python"
            pip_exe = self.shared_env_dir / "bin" / "pip"
            activate_script = self.shared_env_dir / "bin" / "activate"
        
        try:
            # Check if virtual environment already exists and is valid
            if self.python_exe.exists():
                # Test if dependencies are available
                result = subprocess.run([str(self.python_exe), "-c", "import psutil, numpy, pandas, matplotlib"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  Reusing existing virtual environment: {self.shared_env_dir}")
                    return True
                else:
                    print(f"  Virtual environment exists but dependencies missing, recreating...")
                    shutil.rmtree(self.shared_env_dir)
            
            # Create new virtual environment  
            print(f"  Creating virtual environment at: {self.shared_env_dir}")
            self.shared_env_dir.mkdir(exist_ok=True)
            venv.create(self.shared_env_dir, with_pip=True)
            
            # Verify python executable exists
            if not self.python_exe.exists():
                print(f"Error: Python executable not found at {self.python_exe}")
                return False
            
            print("  Upgrading pip...")
            subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], 
                         check=True, capture_output=True, text=True)
            
            # Install all required dependencies
            dependencies = [
                "psutil", "wordcloud", "scikit-learn", "flask", "numpy", "pandas", 
                "matplotlib", "requests", "lxml", "beautifulsoup4", "nltk", "Pillow", 
                "rsa", "faker", "pyquery", "regex", "xlwt", "opencv-python", 
                "tensorflow", "seaborn", "scipy", "networkx", "openpyxl",
                "statsmodels", "soundfile", "flask-mail", "pycryptodome", 
                "gensim", "python-levenshtein", "python-docx", "pytesseract", "chardet"
            ]
            
            print(f"  Installing {len(dependencies)} dependencies...")
            for dep in dependencies:
                try:
                    subprocess.run([str(pip_exe), "install", dep], 
                                 check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"    Warning: Failed to install {dep}: {e}")
            
            # Verify key dependencies are installed
            result = subprocess.run([str(self.python_exe), "-c", 
                                   "import psutil, numpy, pandas, matplotlib, sklearn; print('Dependencies verified')"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Virtual environment ready: {self.shared_env_dir}")
                print(f"  Activate with: source {activate_script}")
                return True
            else:
                print(f"  Error: Dependency verification failed")
                print(f"  stderr: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            return False
    
    def cleanup_shared_environment(self):
        """Clean up the shared environment."""
        if hasattr(self, 'shared_env_dir') and self.shared_env_dir and self.shared_env_dir.exists():
            shutil.rmtree(self.shared_env_dir)
            print(f"Virtual environment cleaned up: {self.shared_env_dir}")
    
    def prepare_test_file(self, temp_dir: Path, task_data: Dict, generated_code: str) -> Path:
        """Prepare a complete test file with generated code and test cases."""
        test_file = temp_dir / "test_task.py"
        
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
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_content))
        
        return test_file
    
    def run_test_in_env(self, temp_dir: Path, test_file: Path) -> Tuple[bool, str, str]:
        """Run the test file in the shared environment."""
        try:
            # Change to temp directory for test execution
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            # Debug: Check if python_exe is set
            if self.python_exe is None:
                return False, "", "Python executable is None"
                
            # Run the test using the shared python environment
            result = subprocess.run(
                [str(self.python_exe), str(test_file)],
                capture_output=True,
                text=True,
                timeout=20  # Reduced to 10 second timeout per test
            )
            
            os.chdir(original_cwd)
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            return False, "", "Test timed out after 30 seconds"
        except Exception as e:
            os.chdir(original_cwd)
            return False, "", f"Error running test: {str(e)}"
    
    def get_requirements_from_libs(self, libs: List[str]) -> List[str]:
        """Convert library names to pip package names."""
        # Common library to package mappings
        lib_to_package = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'wordcloud': 'wordcloud',
            'psutil': 'psutil',
            'flask': 'flask',
            'flask_login': 'flask-login',
            'flask_wtf': 'flask-wtf',
            'wtforms': 'wtforms',
            'werkzeug': 'werkzeug',
            'seaborn': 'seaborn',
            'matplotlib': 'matplotlib',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scipy': 'scipy',
            'requests': 'requests',
            'beautifulsoup4': 'beautifulsoup4',
            'lxml': 'lxml',
            'openpyxl': 'openpyxl',
            'xlsxwriter': 'xlsxwriter'
        }
        
        requirements = []
        for lib in libs:
            if lib in lib_to_package:
                requirements.append(lib_to_package[lib])
            elif lib not in ['os', 'sys', 'json', 'csv', 'sqlite3', 'subprocess', 
                           'threading', 'multiprocessing', 'time', 'datetime', 
                           'collections', 'itertools', 'functools', 'operator',
                           'random', 'math', 'statistics', 're', 'string',
                           'pathlib', 'glob', 'shutil', 'tempfile', 'zipfile',
                           'unittest', 'unittest.mock']:
                # Add non-standard library packages
                requirements.append(lib)
        
        return requirements
    
    def test_single_task(self, task_id: str, task_data: Dict, generated_code: str) -> Dict:
        """Test a single generated code against its original test case."""
        print(f"Testing {task_id}...")
        
        result = {
            'id': task_id,
            'test_result': 0,
            'failure_reason': None
        }
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            try:
                # Ensure shared environment is set up
                if not self.setup_shared_environment():
                    result['failure_reason'] = "Failed to create shared environment"
                    return result
                
                # Debug: Check if python_exe is set
                if self.python_exe is None:
                    result['failure_reason'] = "Python executable not set after environment setup"
                    return result
                
                # Prepare test file
                test_file = self.prepare_test_file(temp_dir, task_data, generated_code)
                
                # Run test
                success, stdout, stderr = self.run_test_in_env(temp_dir, test_file)
                
                if success:
                    result['test_result'] = 1
                    result['failure_reason'] = None
                    print(f"  ✅ PASSED")
                else:
                    result['test_result'] = 0
                    # Extract meaningful failure reason from stderr
                    if stderr:
                        # Look for specific error patterns
                        if "ModuleNotFoundError" in stderr:
                            module_match = stderr.split("ModuleNotFoundError: No module named '")[1].split("'")[0] if "ModuleNotFoundError: No module named '" in stderr else "unknown"
                            result['failure_reason'] = f"Missing module: {module_match}"
                        elif "TimeoutError" in stderr or "timed out" in stderr.lower():
                            result['failure_reason'] = "Test timeout"
                        elif "FAIL:" in stderr:
                            # Extract first test failure
                            fail_lines = [line for line in stderr.split('\n') if 'FAIL:' in line]
                            if fail_lines:
                                result['failure_reason'] = f"Test failure: {fail_lines[0].split('FAIL:')[1].strip()}"
                            else:
                                result['failure_reason'] = "Test assertion failed"
                        elif "ERROR:" in stderr:
                            # Extract first test error
                            error_lines = [line for line in stderr.split('\n') if 'ERROR:' in line]
                            if error_lines:
                                result['failure_reason'] = f"Test error: {error_lines[0].split('ERROR:')[1].strip()}"
                            else:
                                result['failure_reason'] = "Test execution error"
                        elif "SyntaxError" in stderr:
                            result['failure_reason'] = "Syntax error in generated code"
                        elif "IndentationError" in stderr:
                            result['failure_reason'] = "Indentation error in generated code"
                        elif "NameError" in stderr:
                            result['failure_reason'] = "Name error in generated code"
                        elif "AttributeError" in stderr:
                            result['failure_reason'] = "Attribute error in generated code"
                        else:
                            # Truncate long error messages
                            error_summary = stderr.strip()[:200].replace('\n', ' ')
                            result['failure_reason'] = f"Runtime error: {error_summary}"
                    else:
                        result['failure_reason'] = "Unknown test failure"
                    
                    print(f"  ❌ FAILED: {result['failure_reason']}")
                
            except Exception as e:
                result['test_result'] = 0
                result['failure_reason'] = f"Framework error: {str(e)}"
                print(f"  💥 ERROR: {str(e)}")
        
        return result
    
    def test_generated_file(self, file_path: Path) -> List[Dict]:
        """Test all generated code in a file against original test cases."""
        print(f"\n{'='*60}")
        print(f"Testing file: {file_path.name}")
        print(f"{'='*60}")
        
        # Load dataset and generated code
        dataset = self.load_dataset()
        generated_codes = self.load_generated_code(file_path)
        
        results = []
        
        for i, generated_item in enumerate(generated_codes):
            task_id = generated_item['task_id']
            print(f"Progress: {i+1}/{len(generated_codes)} tasks")
            
            # For bigcode format, use original_task_id for dataset lookup
            lookup_id = generated_item.get('original_task_id', task_id)
            
            if lookup_id not in dataset:
                print(f"Warning: Task {lookup_id} not found in dataset (original task_id: {task_id})")
                # Still add to results with failure
                results.append({
                    'id': task_id,
                    'test_result': 0,
                    'failure_reason': 'Task not found in dataset'
                })
                continue
            
            task_data = dataset[lookup_id]
            generated_code = generated_item.get('response_code', '')
            
            if not generated_code:
                print(f"Warning: No generated code for {task_id}")
                results.append({
                    'id': task_id,
                    'test_result': 0,
                    'failure_reason': 'No generated code found'
                })
                continue
            
            # Test this specific task
            result = self.test_single_task(task_id, task_data, generated_code)
            results.append(result)
        
        return results
    
    def save_file_results(self, file_name: str, results: List[Dict]) -> None:
        """Save results for a single file in the required format."""
        # Create output directory if it doesn't exist
        results_dir = self.output_dir
        results_dir.mkdir(exist_ok=True)
        
        # Save results as JSONL (only save here, pipeline will move to model directory)
        output_file = results_dir / file_name
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Results saved to: {output_file}")
    
    def test_all_generated_files(self) -> Dict[str, List[Dict]]:
        """Test all generated code files."""
        all_results = {}
        
        # Find all generated code files
        code_files = list(self.generated_code_dir.glob("code_*.jsonl"))
        
        if not code_files:
            print("No generated code files found!")
            return all_results
        
        print(f"Found {len(code_files)} generated code files")
        
        for file_path in sorted(code_files):
            print(f"\nProcessing {file_path.name}...")
            file_results = self.test_generated_file(file_path)
            all_results[file_path.name] = file_results
            
            # Save results immediately for this file
            self.save_file_results(file_path.name, file_results)
            
            # Print summary for this file
            total = len(file_results)
            passed = sum(1 for r in file_results if r['test_result'] == 1)
            print(f"File {file_path.name}: {passed}/{total} passed ({passed/total*100:.1f}%)")
        
        return all_results
    
    def generate_summary_report(self, all_results: Dict[str, List[Dict]]) -> None:
        """Generate a summary report of all test results."""
        print(f"\n{'='*80}")
        print("FINAL SUMMARY REPORT")
        print(f"{'='*80}")
        
        total_files = len(all_results)
        total_tasks = 0
        total_passed = 0
        
        file_summaries = {}
        
        for file_name, file_results in all_results.items():
            file_total = len(file_results)
            file_passed = sum(1 for r in file_results if r['test_result'] == 1)
            
            file_summaries[file_name] = {
                'total': file_total,
                'passed': file_passed,
                'failed': file_total - file_passed,
                'pass_rate': file_passed / file_total * 100 if file_total > 0 else 0
            }
            
            total_tasks += file_total
            total_passed += file_passed
            
            print(f"\n{file_name}:")
            print(f"  Total: {file_total}, Passed: {file_passed}, Failed: {file_total - file_passed}")
            print(f"  Pass Rate: {file_passed/file_total*100:.1f}%")
        
        print(f"\nOVERALL SUMMARY:")
        print(f"Files tested: {total_files}")
        print(f"Tasks tested: {total_tasks}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tasks - total_passed}")
        print(f"Overall Pass Rate: {total_passed/total_tasks*100:.1f}%")
        
        # Save summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'files_tested': total_files,
                'total_tasks': total_tasks,
                'total_passed': total_passed,
                'total_failed': total_tasks - total_passed,
                'overall_pass_rate': total_passed/total_tasks*100 if total_tasks > 0 else 0,
                'file_summaries': file_summaries
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary saved to: {summary_file}")
        print(f"Individual file results saved to: {self.output_dir}/ directory")


def main():
    """Main function to run the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test generated code against original test cases")
    parser.add_argument("--dataset", default="dataset/bigcodebench.jsonl", 
                       help="Path to the original dataset")
    parser.add_argument("--generated", default="generation_output/o4_standardized",
                       help="Directory containing generated code files")
    parser.add_argument("--file", help="Test only a specific generated code file")
    parser.add_argument("--output-dir", default="test_result",
                       help="Directory to save test results")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = IsolatedTestRunner(args.dataset, args.generated, args.output_dir)
    
    if args.file:
        # Test specific file
        file_path = Path(args.generated) / args.file
        if not file_path.exists():
            print(f"Error: File {file_path} not found")
            return
        
        results = runner.test_generated_file(file_path)
        
        # Save results for single file
        runner.save_file_results(args.file, results)
        
        # Simple summary for single file
        total = len(results)
        passed = sum(1 for r in results if r['test_result'] == 1)
        print(f"\nSingle File Summary:")
        print(f"Total: {total}, Passed: {passed}, Failed: {total-passed}")
        print(f"Pass Rate: {passed/total*100:.1f}%")
        
    else:
        # Test all files - this is the main automation
        all_results = runner.test_all_generated_files()
        runner.generate_summary_report(all_results)


if __name__ == "__main__":
    main()