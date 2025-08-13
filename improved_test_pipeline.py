#!/usr/bin/env python3
"""
Improved pipeline for testing generated code against original test cases.
Includes better error handling, validation, and detailed reporting.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any


class ImprovedTestGeneratedCodePipeline:
    """Improved pipeline for testing generated code with comprehensive reporting."""
    
    def __init__(self, model_name: str, base_dir: str = "generation_output"):
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / model_name
        self.standardized_dir = self.base_dir / f"{model_name}_standardized"
        self.actual_model_name = model_name
        
    def extract_actual_model_name(self) -> str:
        """Extract the actual model name from filenames."""
        code_files = list(self.model_dir.glob("code_*.jsonl"))
        if not code_files:
            return self.model_name
        
        # Extract model name from first file
        filename = code_files[0].name
        parts = filename.replace('.jsonl', '').split('_')
        
        # Find the model name (part that comes before 'bigcodebench')
        for i, part in enumerate(parts):
            if part == 'bigcodebench' and i > 0:
                actual_model_name = parts[i-1]
                print(f"Auto-detected model name: {actual_model_name}")
                return actual_model_name
        
        return self.model_name
    
    def validate_directories(self) -> bool:
        """Validate that required directories exist."""
        if not self.model_dir.exists():
            print(f"Error: Model directory {self.model_dir} does not exist!")
            return False
        
        # Check if we have any code files
        code_files = list(self.model_dir.glob("code_*.jsonl"))
        if not code_files:
            print(f"Error: No code_*.jsonl files found in {self.model_dir}")
            return False
        
        # Extract actual model name from files
        self.actual_model_name = self.extract_actual_model_name()
        if self.actual_model_name != self.model_name:
            print(f"Note: Using actual model name '{self.actual_model_name}' instead of '{self.model_name}'")
            # Update directories to use actual model name
            self.standardized_dir = self.base_dir / f"{self.actual_model_name}_standardized"
            
        print(f"Found {len(code_files)} code files in {self.model_dir}")
        return True
    
    def run_improved_standardization(self) -> bool:
        """Run the improved standardization process."""
        print("Step 1: Standardizing function names (improved)...")
        
        try:
            result = subprocess.run([
                sys.executable, 
                "improved_standardize_function_names.py",
                "--input-dir", str(self.model_dir),
                "--output-dir", str(self.standardized_dir),
                "--validate"
            ], capture_output=True, text=True, check=True)
            
            print("Improved function name standardization completed successfully!")
            print("Standardization output:")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error in improved function name standardization: {e}")
            print(f"stderr: {e.stderr}")
            return False
    
    def run_improved_test_generation(self, test_file: str = None, workers: int = None) -> bool:
        """Run the improved test generation process with multithreading."""
        print("Step 2: Running improved test generation with multithreading...")
        
        cmd = [
            sys.executable, 
            "improved_test_generated_code.py",
            "--dataset", "dataset/bigcodebench.jsonl",
            "--generated", str(self.standardized_dir),
            "--output-dir", "test_result"
        ]
        
        if test_file:
            cmd.extend(["--file", test_file])
            print(f"  Testing only file: {test_file}")
        
        if workers:
            cmd.extend(["--workers", str(workers)])
            print(f"  Using {workers} worker threads")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=7200)  # 2 hour timeout
            
            print("Improved test generation completed successfully!")
            print("Test output:")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error in improved test generation: {e}")
            print(f"stderr: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("Test generation timed out after 2 hours")
            return False
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and provide detailed statistics."""
        print("Step 3: Analyzing results...")
        
        model_name = self.actual_model_name
        results_dir = Path("test_result")
        
        # Find result files for this model
        result_files = list(results_dir.glob(f"code_*_{model_name}_*.jsonl"))
        
        if not result_files:
            print(f"No result files found for model {model_name}")
            return {}
        
        analysis = {
            "model_name": model_name,
            "total_files": len(result_files),
            "file_results": {},
            "overall_stats": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "error_breakdown": {}
            }
        }
        
        for result_file in result_files:
            print(f"  Analyzing: {result_file.name}")
            
            file_stats = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "error_types": {}
            }
            
            try:
                with open(result_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line.strip())
                            file_stats["total"] += 1
                            
                            if result["test_result"] == 1:
                                file_stats["passed"] += 1
                            else:
                                file_stats["failed"] += 1
                                failure_reason = result.get("failure_reason", "Unknown")
                                
                                # Categorize error types
                                if "Syntax error" in failure_reason:
                                    error_type = "Syntax Error"
                                elif "Missing module" in failure_reason:
                                    error_type = "Missing Module"
                                elif "Test failure" in failure_reason:
                                    error_type = "Test Assertion Failure"
                                elif "Test error" in failure_reason:
                                    error_type = "Test Execution Error"
                                elif "Test timeout" in failure_reason:
                                    error_type = "Timeout"
                                elif "Runtime error" in failure_reason:
                                    error_type = "Runtime Error"
                                else:
                                    error_type = "Other"
                                
                                file_stats["error_types"][error_type] = file_stats["error_types"].get(error_type, 0) + 1
                
                file_stats["pass_rate"] = file_stats["passed"] / file_stats["total"] * 100 if file_stats["total"] > 0 else 0
                analysis["file_results"][result_file.name] = file_stats
                
                # Update overall stats
                analysis["overall_stats"]["total_tests"] += file_stats["total"]
                analysis["overall_stats"]["passed_tests"] += file_stats["passed"]
                analysis["overall_stats"]["failed_tests"] += file_stats["failed"]
                
                for error_type, count in file_stats["error_types"].items():
                    analysis["overall_stats"]["error_breakdown"][error_type] = analysis["overall_stats"]["error_breakdown"].get(error_type, 0) + count
                
            except Exception as e:
                print(f"    Error analyzing {result_file.name}: {e}")
        
        # Calculate overall pass rate
        total = analysis["overall_stats"]["total_tests"]
        passed = analysis["overall_stats"]["passed_tests"]
        analysis["overall_stats"]["pass_rate"] = passed / total * 100 if total > 0 else 0
        
        return analysis
    
    def save_analysis_report(self, analysis: Dict[str, Any]) -> bool:
        """Save detailed analysis report."""
        try:
            model_name = self.actual_model_name
            results_dir = Path("test_result")
            report_file = results_dir / f"{model_name}_detailed_analysis.json"
            
            with open(report_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"Detailed analysis saved to: {report_file}")
            return True
            
        except Exception as e:
            print(f"Error saving analysis report: {e}")
            return False
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print a summary of the analysis."""
        if not analysis:
            return
        
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {analysis['model_name']}")
        print(f"Files analyzed: {analysis['total_files']}")
        
        overall = analysis["overall_stats"]
        print(f"\nOverall Performance:")
        print(f"  Total tests: {overall['total_tests']}")
        print(f"  Passed: {overall['passed_tests']}")
        print(f"  Failed: {overall['failed_tests']}")
        print(f"  Pass rate: {overall['pass_rate']:.1f}%")
        
        if overall["error_breakdown"]:
            print(f"\nError Breakdown:")
            for error_type, count in sorted(overall["error_breakdown"].items(), key=lambda x: x[1], reverse=True):
                percentage = count / overall['failed_tests'] * 100 if overall['failed_tests'] > 0 else 0
                print(f"  {error_type}: {count} ({percentage:.1f}% of failures)")
        
        print(f"\nPer-file Results:")
        for filename, stats in analysis["file_results"].items():
            print(f"  {filename}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1f}%)")
    
    def run_pipeline(self, test_file: str = None, workers: int = None) -> bool:
        """Run the complete improved pipeline with multithreading."""
        print(f"{'='*80}")
        print(f"Running Improved Test Generated Code Pipeline for model: {self.model_name}")
        if workers:
            print(f"Using {workers} worker threads for parallel testing")
        print(f"{'='*80}")
        
        if not self.validate_directories():
            return False
        
        # Step 1: Improved standardization
        if not self.run_improved_standardization():
            return False
        
        # Step 2: Improved test generation with multithreading
        if not self.run_improved_test_generation(test_file, workers):
            return False
        
        # Step 3: Analyze results
        analysis = self.analyze_results()
        
        # Step 4: Save and print analysis
        if analysis:
            self.save_analysis_report(analysis)
            self.print_analysis_summary(analysis)
        
        model_name = self.actual_model_name
        print(f"\n{'='*80}")
        print("IMPROVED MULTITHREADED PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Results saved to: test_result/")
        print(f"Detailed analysis: test_result/{model_name}_detailed_analysis.json")
        print(f"{'='*80}")
        
        return True


def main():
    """Main function."""
    import multiprocessing
    
    parser = argparse.ArgumentParser(description="Improved Test Generated Code Pipeline with Multithreading")
    parser.add_argument("--model", required=True, 
                       help="Model name (e.g., 'o4-mini')")
    parser.add_argument("--base-dir", default="generation_output",
                       help="Base directory containing model outputs")
    parser.add_argument("--test-file", 
                       help="Test only a specific file")
    parser.add_argument("--workers", type=int, default=None,
                       help=f"Number of worker threads (default: min(CPU_count, 8) = {min(multiprocessing.cpu_count(), 8)})")
    parser.add_argument("--sequential", action="store_true",
                       help="Run tests sequentially instead of in parallel (for debugging)")
    
    args = parser.parse_args()
    
    # Set workers based on args
    if args.sequential:
        workers = 1
        print("Running in sequential mode for debugging")
    else:
        workers = args.workers or min(multiprocessing.cpu_count(), 8)
        print(f"Using {workers} worker threads for parallel testing")
    
    # Initialize and run pipeline
    pipeline = ImprovedTestGeneratedCodePipeline(args.model, args.base_dir)
    success = pipeline.run_pipeline(args.test_file, workers)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()