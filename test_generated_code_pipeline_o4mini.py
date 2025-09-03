#!/usr/bin/env python3
"""
Pipeline for testing generated code against original test cases.
Adapted for o4-mini format in generation_output3/.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import time


class TestGeneratedCodePipelineO4Mini:
    """Pipeline for testing generated code with o4-mini format."""
    
    def __init__(self, model_name: str = "o4-mini", base_dir: str = "generation_output3", resume: bool = False):
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / model_name
        self.resume = resume
        
        # Check if using bigcode subdirectory structure
        bigcode_subdir = self.base_dir / "bigcode" / model_name
        if bigcode_subdir.exists():
            self.code_dir = bigcode_subdir
            self.use_bigcode_subdir = True
            self.use_bigcode_format = True
        else:
            self.code_dir = self.model_dir
            self.use_bigcode_subdir = False
            self.use_bigcode_format = False
            
        self.standardized_dir = self.base_dir / f"{model_name}_standardized"
        
    def extract_actual_model_name(self) -> str:
        """Extract the actual model name from filenames."""
        code_files = list(self.code_dir.glob("code_*.jsonl"))
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
            # O4-mini format: code_{technique}_{model_name}_bigcodebench_output_llm{X}[_v{Y}].jsonl
            # Find the model name (part that comes before 'bigcodebench')
            for i, part in enumerate(parts):
                if part == 'bigcodebench' and i > 0:
                    actual_model_name = parts[i-1]
                    print(f"Auto-detected model name (o4-mini format): {actual_model_name}")
                    return actual_model_name
        
        return self.model_name
    
    def validate_directories(self) -> bool:
        """Validate that required directories exist."""
        if not self.model_dir.exists() and not (self.base_dir / "bigcode" / self.model_name).exists():
            print(f"Error: Model directory {self.model_dir} does not exist!")
            return False
        
        if not self.code_dir.exists():
            print(f"Error: Code directory {self.code_dir} does not exist!")
            return False
        
        # Check if we have any code files
        code_files = list(self.code_dir.glob("code_*.jsonl"))
        if not code_files:
            print(f"Error: No code_*.jsonl files found in {self.code_dir}")
            return False
        
        # Extract actual model name from files
        self.actual_model_name = self.extract_actual_model_name()
        if self.actual_model_name != self.model_name:
            print(f"Note: Using actual model name '{self.actual_model_name}' instead of '{self.model_name}'")
            # Update directories to use actual model name
            self.standardized_dir = self.base_dir / f"{self.actual_model_name}_standardized"
        
        if self.use_bigcode_subdir:
            print(f"Using bigcode/ subdirectory structure: Found {len(code_files)} code files in {self.code_dir}")
        else:
            print(f"Using flat directory structure: Found {len(code_files)} code files in {self.code_dir}")
            
        # Group files by technique for summary
        techniques = set()
        for file in code_files:
            parts = file.name.replace('.jsonl', '').split('_')
            if len(parts) >= 4:  # code_{technique}_{model}_bigcodebench_...
                technique = parts[1]
                techniques.add(technique)
        print(f"  - techniques: {sorted(techniques)}")
            
        return True
    
    def check_standardization_completed(self) -> bool:
        """Check if standardization has already been completed."""
        if not self.standardized_dir.exists():
            return False
        
        # Check if standardized directory has files
        standardized_files = list(self.standardized_dir.glob("code_*.jsonl"))
        original_files = list(self.code_dir.glob("code_*.jsonl"))
        
        return len(standardized_files) > 0 and len(standardized_files) >= len(original_files)
    
    def run_standardization(self) -> bool:
        """Run the standardization process."""
        if self.resume and self.check_standardization_completed():
            print("Step 1: Standardization already completed, skipping...")
            return True
            
        print("Step 1: Standardizing function names...")
        
        # Find the standardize_function_names.py script
        script_path = Path(__file__).parent / "standardize_function_names.py"
        if not script_path.exists():
            # Try different locations
            possible_paths = [
                Path("/Users/aliredaq/Downloads/bigcodebench/standardize_function_names.py"),
                Path("../standardize_function_names.py"),
                Path("../../standardize_function_names.py")
            ]
            for path in possible_paths:
                if path.exists():
                    script_path = path
                    break
            else:
                print("Error: standardize_function_names.py not found!")
                return False
        
        # Run standardize_function_names.py with modified paths
        try:
            with tqdm(desc="Standardizing function names", unit="step") as pbar:
                pbar.set_description("Running standardization...")
                result = subprocess.run([
                    sys.executable, 
                    str(script_path),
                    "--input-dir", str(self.code_dir),
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
    
    def check_test_generation_completed(self, test_file: str = None) -> bool:
        """Check if test generation has already been completed."""
        results_dir = Path("/Users/aliredaq/Downloads/bigcodebench/test_result5")
        if not results_dir.exists():
            return False
        
        # Check both in main results dir and model-specific results dir
        # Ensure actual_model_name is set
        if not hasattr(self, 'actual_model_name'):
            self.actual_model_name = self.extract_actual_model_name()
        model_name = getattr(self, 'actual_model_name', self.model_name)
        model_results_dir = results_dir / f"{model_name}_results"
        
        if test_file:
            # Check for specific file in both locations
            expected_result_main = results_dir / test_file
            expected_result_model = model_results_dir / test_file
            return expected_result_main.exists() or expected_result_model.exists()
        else:
            # Check if we have results for all files
            result_files_main = list(results_dir.glob("code_*.jsonl"))
            result_files_model = list(model_results_dir.glob("code_*.jsonl")) if model_results_dir.exists() else []
            
            total_result_files = len(result_files_main) + len(result_files_model)
            standardized_files = list(self.standardized_dir.glob("code_*.jsonl"))
            return total_result_files > 0 and total_result_files >= len(standardized_files)
    
    def get_missing_test_files(self) -> List[str]:
        """Get list of files that need test generation."""
        if not hasattr(self, 'actual_model_name'):
            self.actual_model_name = self.extract_actual_model_name()
        model_name = getattr(self, 'actual_model_name', self.model_name)
        
        # Get all standardized files
        standardized_files = list(self.standardized_dir.glob("code_*.jsonl"))
        
        # Get existing result files - use absolute path
        results_dir = Path("/Users/aliredaq/Downloads/bigcodebench/test_result5")
        model_results_dir = results_dir / f"{model_name}_results"
        
        existing_results = set()
        if results_dir.exists():
            existing_results.update([f.name for f in results_dir.glob("code_*.jsonl")])
        if model_results_dir.exists():
            existing_results.update([f.name for f in model_results_dir.glob("code_*.jsonl")])
        
        # Find missing files by checking if corresponding result exists
        missing_files = []
        for file in standardized_files:
            # Check if result exists with exact name
            if file.name in existing_results:
                continue
                
            # Check if result exists with transformed name using direct pattern matching
            has_result = False
            filename_parts = file.name.replace('.jsonl', '').split('_')
            if len(filename_parts) >= 4:
                technique = filename_parts[1]  # e.g., "active", "adversarial", etc.
                
                # Look for results that start with the same technique
                for result_name in existing_results:
                    result_parts = result_name.replace('.jsonl', '').split('_')
                    if (len(result_parts) >= 3 and 
                        result_parts[1] == technique and  # Same technique
                        result_parts[2] == model_name):   # Has model name in position 2
                        has_result = True
                        break
            
            if not has_result:
                missing_files.append(file.name)
        
        # Debug: show what we found
        if self.resume and missing_files:
            print(f"Debug: Found {len(existing_results)} existing results:")
            for r in sorted(list(existing_results))[:3]:
                print(f"  - {r}")
            if len(existing_results) > 3:
                print(f"  ... and {len(existing_results) - 3} more")
        
        return missing_files
    
    def run_test_generation(self, test_file: str = None) -> bool:
        """Run the test generation process."""
        if self.resume:
            if test_file:
                # Check specific file
                if self.check_test_generation_completed(test_file):
                    print(f"Step 2: Test generation for {test_file} already completed, skipping...")
                    return True
            else:
                # Check what files are missing
                missing_files = self.get_missing_test_files()
                if not missing_files:
                    print("Step 2: Test generation already completed, skipping...")
                    return True
                else:
                    print(f"Step 2: Found {len(missing_files)} files that need test generation:")
                    for f in missing_files[:5]:  # Show first 5
                        print(f"  - {f}")
                    if len(missing_files) > 5:
                        print(f"  ... and {len(missing_files) - 5} more")
                    
                    # Process only the missing files one by one
                    print("Step 2: Processing only missing files individually...")
                    success_count = 0
                    for missing_file in missing_files:
                        print(f"\nProcessing: {missing_file}")
                        file_success = self._run_single_test_generation(missing_file)
                        if file_success:
                            success_count += 1
                        else:
                            print(f"Failed to process {missing_file}")
                    
                    print(f"Completed {success_count}/{len(missing_files)} files successfully")
                    return success_count > 0  # Return True if at least one file succeeded
            
        # Fall back to processing all files if not in resume mode
        return self._run_all_test_generation(test_file)
    
    def _run_single_test_generation(self, test_file: str) -> bool:
        """Run test generation for a single file."""
        return self._run_all_test_generation(test_file)
    
    def _run_all_test_generation(self, test_file: str = None) -> bool:
        """Run test generation for all files or a specific file."""
        print("Step 2: Running test generation...")
        
        # Find the test_generated_code.py script
        test_script_path = Path(__file__).parent / "test_generated_code.py"
        if not test_script_path.exists():
            test_script_path = Path("/Users/aliredaq/Downloads/bigcodebench/test_generated_code.py")
        
        # Find the dataset file
        dataset_path = Path(__file__).parent / "dataset" / "bigcodebench.jsonl"
        if not dataset_path.exists():
            dataset_path = Path("/Users/aliredaq/Downloads/bigcodebench/dataset/bigcodebench.jsonl")
        
        cmd = [
            sys.executable, 
            str(test_script_path),
            "--dataset", str(dataset_path),
            "--generated", str(self.standardized_dir),
            "--output-dir", "/Users/aliredaq/Downloads/bigcodebench/test_result5"
        ]
        
        if test_file:
            cmd.extend(["--file", test_file])
            print(f"  Testing only file: {test_file}")
        
        try:
            with tqdm(desc="Running test generation", unit="step") as pbar:
                pbar.set_description("Executing tests...")
                print(f"Running command: {' '.join(cmd)}")
                # Don't capture output so we can see what's happening
                result = subprocess.run(cmd, timeout=3600)  # 1 hour timeout
                pbar.update(1)
                pbar.set_description("Test generation completed")
            
            if result.returncode == 0:
                print("Test generation completed successfully!")
                return True
            else:
                print(f"Test generation failed with return code: {result.returncode}")
                return False
            
        except subprocess.TimeoutExpired as e:
            print(f"Test generation timed out after 1 hour: {e}")
            print("Treating timeout as failure and continuing...")
            return False
        except Exception as e:
            print(f"Error in test generation: {e}")
            return False
    
    def check_results_organization_completed(self) -> bool:
        """Check if results organization has already been completed."""
        results_dir = Path("/Users/aliredaq/Downloads/bigcodebench/test_result5")
        if not results_dir.exists():
            return False
        
        # Ensure actual_model_name is set
        if not hasattr(self, 'actual_model_name'):
            self.actual_model_name = self.extract_actual_model_name()
        model_name = getattr(self, 'actual_model_name', self.model_name)
        model_results_dir = results_dir / f"{model_name}_results"
        
        # Check if model results directory exists and has files
        if not model_results_dir.exists():
            return False
            
        result_files = list(model_results_dir.glob("code_*.jsonl"))
        return len(result_files) > 0
    
    def create_model_specific_results(self) -> bool:
        """Create model-specific results directory and rename files."""
        if self.resume and self.check_results_organization_completed():
            print("Step 3: Results organization already completed, skipping...")
            return True
            
        print("Step 3: Creating model-specific results...")
        
        # Use actual model name for results
        model_name = getattr(self, 'actual_model_name', self.model_name)
        
        # Create model-specific results directory
        results_dir = Path("/Users/aliredaq/Downloads/bigcodebench/test_result5")
        model_results_dir = results_dir / f"{model_name}_results"
        model_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Move and rename result files to include model name
        if results_dir.exists():
            result_files = list(results_dir.glob("code_*.jsonl"))
            with tqdm(result_files, desc="Organizing results") as pbar:
                for result_file in pbar:
                    pbar.set_description(f"Processing {result_file.name}")
                    filename_parts = result_file.name.replace('.jsonl', '').split('_')
                    
                    # Handle the o4-mini format: code_{technique}_{model_name}_bigcodebench_output_llm{X}[_v{Y}].jsonl
                    if len(filename_parts) >= 4:
                        technique = filename_parts[1]  # e.g., "original", "active", etc.
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
        print(f"Running Test Generated Code Pipeline for o4-mini")
        if self.resume:
            print("(Resume mode: skipping completed steps)")
        print(f"Base directory: {self.base_dir}")
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
    parser = argparse.ArgumentParser(description="Test Generated Code Pipeline for O4-Mini")
    parser.add_argument("--model", default="o4-mini", 
                       help="Model name (default: 'o4-mini')")
    parser.add_argument("--base-dir", default="generation_output3",
                       help="Base directory containing model outputs (default: generation_output3)")
    parser.add_argument("--test-file", 
                       help="Test only a specific file (e.g., 'code_original_o4-mini_bigcodebench_output_llm1.jsonl')")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run, skipping completed steps")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = TestGeneratedCodePipelineO4Mini(args.model, args.base_dir, args.resume)
    success = pipeline.run_pipeline(args.test_file)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()