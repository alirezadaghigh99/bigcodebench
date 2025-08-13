#!/usr/bin/env python3
"""
Pipeline for testing generated code against original test cases.
Includes model name parameter and standardization steps.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import time


class TestGeneratedCodePipeline:
    """Pipeline for testing generated code with model name parameter."""
    
    def __init__(self, model_name: str, base_dir: str = "generation_output"):
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        # Check if using new bigcode structure
        bigcode_dir = self.base_dir / model_name
        if bigcode_dir.exists():
            self.model_dir = bigcode_dir
            self.use_bigcode_format = True
        else:
            self.model_dir = self.base_dir / model_name
            self.use_bigcode_format = False
        self.standardized_dir = self.base_dir / f"{model_name}_standardized"
        
    def extract_actual_model_name(self) -> str:
        """Extract the actual model name from filenames."""
        code_files = list(self.model_dir.glob("code_*.jsonl"))
        if not code_files:
            return self.model_name
        
        # Extract model name from first file
        filename = code_files[0].name
        parts = filename.replace('.jsonl', '').split('_')
        
        if self.use_bigcode_format:
            # Bigcode format: code_{technique}_{model_name}_bigcodebench_v{X}.jsonl
            # Find the model name (part that comes before 'bigcodebench')
            for i, part in enumerate(parts):
                if part == 'bigcodebench' and i > 0:
                    actual_model_name = parts[i-1]
                    print(f"Auto-detected model name (bigcode format): {actual_model_name}")
                    return actual_model_name
        else:
            # Current format: code_{technique}_{model_name}_bigcodebench_output_llm{X}[_v{Y}].jsonl
            # Find the model name (part that comes before 'bigcodebench')
            for i, part in enumerate(parts):
                if part == 'bigcodebench' and i > 0:
                    actual_model_name = parts[i-1]
                    print(f"Auto-detected model name (current format): {actual_model_name}")
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
        
        if self.use_bigcode_format:
            print(f"Using bigcode format: Found {len(code_files)} code files in {self.model_dir}")
            # Group files by technique and version for summary
            techniques = {}
            for file in code_files:
                parts = file.name.replace('.jsonl', '').split('_')
                if len(parts) >= 4 and parts[-1].startswith('v'):
                    technique = parts[1]  # e.g., 'original' from 'code_original_model_bigcodebench_v1.jsonl'
                    version = parts[-1]   # e.g., 'v1'
                    if technique not in techniques:
                        techniques[technique] = []
                    techniques[technique].append(version)
            for technique, versions in techniques.items():
                print(f"  - {technique}: {sorted(versions)}")
        else:
            print(f"Using current format: Found {len(code_files)} code files in {self.model_dir}")
            # Group files by technique for summary
            techniques = set()
            for file in code_files:
                parts = file.name.replace('.jsonl', '').split('_')
                if len(parts) >= 4:  # code_{technique}_{model}_bigcodebench_...
                    technique = parts[1]
                    techniques.add(technique)
            print(f"  - techniques: {sorted(techniques)}")
            
        return True
    
    def run_standardization(self) -> bool:
        """Run the standardization process."""
        print("Step 1: Standardizing function names...")
        
        # Run standardize_function_names.py with modified paths
        try:
            with tqdm(desc="Standardizing function names", unit="step") as pbar:
                pbar.set_description("Running standardization...")
                result = subprocess.run([
                    sys.executable, 
                    "standardize_function_names.py",
                    "--input-dir", str(self.model_dir),
                    "--output-dir", str(self.standardized_dir)
                ], capture_output=True, text=True, check=True)
                pbar.update(1)
                pbar.set_description("Standardization completed")
            
            print("Function name standardization completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error in function name standardization: {e}")
            print(f"stderr: {e.stderr}")
            return False
    
    def run_test_generation(self, test_file: str = None) -> bool:
        """Run the test generation process."""
        print("Step 2: Running test generation...")
        
        cmd = [
            sys.executable, 
            "test_generated_code.py",
            "--dataset", "dataset/bigcodebench.jsonl",
            "--generated", str(self.standardized_dir),
            "--output-dir", "test_result5"
        ]
        
        if test_file:
            cmd.extend(["--file", test_file])
            print(f"  Testing only file: {test_file}")
        
        try:
            with tqdm(desc="Running test generation", unit="step") as pbar:
                pbar.set_description("Executing tests...")
                print(f"Running command: {' '.join(cmd)}")
                # Don't capture output so we can see what's happening
                result = subprocess.run(cmd, timeout=3600)  # 1 mins timeout
                pbar.update(1)
                pbar.set_description("Test generation completed")
            
            if result.returncode == 0:
                print("Test generation completed successfully!")
                return True
            else:
                print(f"Test generation failed with return code: {result.returncode}")
                return False
            
        except subprocess.TimeoutExpired as e:
            print(f"Test generation timed out after 60 seconds: {e}")
            print("Treating timeout as failure and continuing...")
            return False
        except Exception as e:
            print(f"Error in test generation: {e}")
            return False
    
    def create_model_specific_results(self) -> bool:
        """Create model-specific results directory and rename files."""
        print("Step 3: Creating model-specific results...")
        
        # Use actual model name for results
        model_name = getattr(self, 'actual_model_name', self.model_name)
        
        # Create model-specific results directory
        results_dir = Path("test_result5")
        model_results_dir = results_dir / f"{model_name}_results"
        model_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Move and rename result files to include model name
        if results_dir.exists():
            result_files = list(results_dir.glob("code_*.jsonl"))
            with tqdm(result_files, desc="Organizing results") as pbar:
                for result_file in pbar:
                    pbar.set_description(f"Processing {result_file.name}")
                    filename_parts = result_file.name.replace('.jsonl', '').split('_')
                    
                    if self.use_bigcode_format:
                        # Bigcode format: code_{technique}_{model_name}_bigcodebench_v{X}.jsonl
                        if len(filename_parts) >= 4 and filename_parts[-1].startswith('v'):
                            technique = filename_parts[1]  # e.g., "original"
                            version = filename_parts[-1]   # e.g., "v1"
                            new_name = f"code_{technique}_{model_name}_bigcodebench_{version}.jsonl"
                        else:
                            new_name = result_file.name
                    else:
                        # Current format: code_{technique}_{model_name}_bigcodebench_output_llm{X}[_v{Y}].jsonl
                        if len(filename_parts) >= 4:
                            technique = filename_parts[1]  # e.g., "original", "active_to_passive", etc.
                            # Keep the original suffix (output_llm1, output_llm1_v2, etc.)
                            suffix_parts = filename_parts[3:]  # everything after model name
                            suffix = '_'.join(suffix_parts)
                            new_name = f"code_{technique}_{model_name}_{suffix}.jsonl"
                        else:
                            new_name = result_file.name
                            
                    new_path = model_results_dir / new_name
                    
                    # Move the file (not copy) to avoid duplicates
                    import shutil
                    shutil.move(result_file, new_path)
                    pbar.set_postfix({"moved": new_path.name})
            
            # Move summary file if it exists
            summary_file = results_dir / "summary.json"
            if summary_file.exists():
                with tqdm(desc="Moving summary", unit="file") as pbar:
                    model_summary = model_results_dir / f"{model_name}_summary.json"
                    import shutil
                    shutil.move(summary_file, model_summary)
                    pbar.update(1)
                    print(f"  Moved: {model_summary}")
        
        return True
    
    def run_pipeline(self, test_file: str = None) -> bool:
        """Run the complete pipeline."""
        print(f"{'='*60}")
        print(f"Running Test Generated Code Pipeline for model: {self.model_name}")
        print(f"{'='*60}")
        
        if not self.validate_directories():
            return False
        
        # Overall progress tracking
        total_steps = 3
        with tqdm(total=total_steps, desc="Pipeline Progress", unit="step") as overall_pbar:
            # Step 1: Standardize function names
            overall_pbar.set_description("Step 1/3: Standardization")
            if not self.run_standardization():
                return False
            overall_pbar.update(1)
            
            # Step 2: Run test generation
            overall_pbar.set_description("Step 2/3: Test Generation")
            test_success = self.run_test_generation(test_file)
            if not test_success:
                print("Test generation failed, but continuing with pipeline...")
            overall_pbar.update(1)
            
            # Step 3: Create model-specific results
            overall_pbar.set_description("Step 3/3: Organizing Results")
            if not self.create_model_specific_results():
                return False
            overall_pbar.update(1)
            
            overall_pbar.set_description("Pipeline Completed")
        
        model_name = getattr(self, 'actual_model_name', self.model_name)
        print(f"\n{'='*60}")
        print("Pipeline completed successfully!")
        print(f"Results saved to: test_result5/{model_name}_results/")
        print(f"{'='*60}")
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Generated Code Pipeline")
    parser.add_argument("--model", required=True, 
                       help="Model name (e.g., 'o4-mini')")
    parser.add_argument("--base-dir", default="generation_output",
                       help="Base directory containing model outputs")
    parser.add_argument("--test-file", 
                       help="Test only a specific file (e.g., 'code_original_o4-mini_bigcodebench_output_llm.jsonl')")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = TestGeneratedCodePipeline(args.model, args.base_dir)
    success = pipeline.run_pipeline(args.test_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()