#!/usr/bin/env python3
"""
BigCodeBench-compatible evaluation pipeline.
Uses the official BigCodeBench evaluation methodology for accurate results.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class BigCodeBenchEvaluator:
    """Wrapper for official BigCodeBench evaluation."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_dir = Path("generation_output")
        self.model_dir = self.base_dir / model_name
        
    def check_bigcodebench_installation(self) -> bool:
        """Check if BigCodeBench is properly installed."""
        try:
            result = subprocess.run([
                sys.executable, "-c", "import bigcodebench; print(bigcodebench.__version__)"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ BigCodeBench installed: {result.stdout.strip()}")
                return True
            else:
                print("❌ BigCodeBench not found")
                return False
        except Exception as e:
            print(f"❌ Error checking BigCodeBench: {e}")
            return False
    
    def install_bigcodebench(self) -> bool:
        """Install BigCodeBench evaluation dependencies."""
        print("📦 Installing BigCodeBench evaluation dependencies...")
        
        try:
            # Install BigCodeBench
            subprocess.run([
                sys.executable, "-m", "pip", "install", "bigcodebench"
            ], check=True)
            
            # Install evaluation requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r",
                "https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt"
            ], check=True)
            
            print("✅ BigCodeBench installed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    def convert_to_bigcodebench_format(self, input_file: Path, output_file: Path) -> bool:
        """Convert our JSONL format to BigCodeBench format."""
        print(f"🔄 Converting {input_file.name} to BigCodeBench format...")
        
        try:
            solutions = []
            with open(input_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        
                        # Extract solution code
                        solution_code = data.get('response_code', '')
                        if not solution_code:
                            # Try to extract from response field
                            response = data.get('response', '')
                            if '```python' in response:
                                # Extract code from markdown
                                import re
                                code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
                                if code_blocks:
                                    solution_code = code_blocks[0]
                        
                        if solution_code:
                            solutions.append({
                                "task_id": data['task_id'],
                                "solution": solution_code.strip()
                            })
            
            # Write in BigCodeBench format
            with open(output_file, 'w') as f:
                for solution in solutions:
                    f.write(json.dumps(solution) + '\n')
            
            print(f"✅ Converted {len(solutions)} solutions")
            return True
            
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return False
    
    def run_official_evaluation(self, solutions_file: Path, execution_mode: str = "local") -> bool:
        """Run the official BigCodeBench evaluation."""
        print(f"🚀 Running official BigCodeBench evaluation...")
        print(f"   Solutions: {solutions_file}")
        print(f"   Execution: {execution_mode}")
        
        try:
            cmd = [
                sys.executable, "-m", "bigcodebench.evaluate",
                "--execution", execution_mode,
                "--split", "complete",
                "--subset", "full",
                "--samples", str(solutions_file)
            ]
            
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
            
            print("✅ Evaluation completed successfully!")
            print("Output:")
            print(result.stdout)
            
            if result.stderr:
                print("Stderr:")
                print(result.stderr)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Evaluation timed out after 1 hour")
            return False
    
    def find_generated_files(self) -> List[Path]:
        """Find generated code files for the model."""
        code_files = []
        
        # Look in standardized directory first
        standardized_dir = self.base_dir / f"{self.model_name}_standardized"
        if standardized_dir.exists():
            code_files.extend(list(standardized_dir.glob("code_*.jsonl")))
        
        # Also look in original directory
        if self.model_dir.exists():
            code_files.extend(list(self.model_dir.glob("code_*.jsonl")))
        
        return sorted(set(code_files))
    
    def evaluate_all_files(self, execution_mode: str = "local") -> Dict[str, Any]:
        """Evaluate all generated files using official BigCodeBench."""
        
        # Check/install BigCodeBench
        if not self.check_bigcodebench_installation():
            if not self.install_bigcodebench():
                return {"error": "Failed to install BigCodeBench"}
        
        # Find generated files
        code_files = self.find_generated_files()
        if not code_files:
            return {"error": f"No code files found for model {self.model_name}"}
        
        print(f"Found {len(code_files)} code files to evaluate")
        
        results = {
            "model_name": self.model_name,
            "execution_mode": execution_mode,
            "files_processed": [],
            "evaluation_results": {},
            "summary": {}
        }
        
        for code_file in code_files:
            print(f"\n{'='*60}")
            print(f"Processing: {code_file.name}")
            print(f"{'='*60}")
            
            # Convert to BigCodeBench format
            bcb_file = Path(f"bcb_{code_file.name}")
            if not self.convert_to_bigcodebench_format(code_file, bcb_file):
                continue
            
            # Run evaluation
            success = self.run_official_evaluation(bcb_file, execution_mode)
            
            results["files_processed"].append({
                "original_file": str(code_file),
                "bcb_file": str(bcb_file),
                "success": success
            })
            
            # Look for result files
            result_files = list(Path(".").glob(f"{bcb_file.stem}_eval_results.json"))
            pass_at_k_files = list(Path(".").glob(f"{bcb_file.stem}_pass_at_k.json"))
            
            if result_files:
                try:
                    with open(result_files[0], 'r') as f:
                        eval_results = json.load(f)
                    results["evaluation_results"][code_file.name] = eval_results
                except Exception as e:
                    print(f"⚠️  Could not load evaluation results: {e}")
            
            if pass_at_k_files:
                try:
                    with open(pass_at_k_files[0], 'r') as f:
                        pass_at_k = json.load(f)
                    results["summary"][code_file.name] = pass_at_k
                    print(f"📊 Results: {pass_at_k}")
                except Exception as e:
                    print(f"⚠️  Could not load pass@k results: {e}")
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Analyze and display evaluation results."""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"\n{'='*80}")
        print("OFFICIAL BIGCODEBENCH EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Model: {results['model_name']}")
        print(f"Execution Mode: {results['execution_mode']}")
        print(f"Files Processed: {len(results['files_processed'])}")
        
        if results['summary']:
            print(f"\n📊 Pass@K Results:")
            for filename, pass_results in results['summary'].items():
                print(f"\n{filename}:")
                for metric, value in pass_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.3f}")
                    else:
                        print(f"  {metric}: {value}")
        
        # Compare with expected performance
        if results['summary']:
            # Get pass@1 for the main file (original)
            original_files = [k for k in results['summary'].keys() if 'original' in k]
            if original_files:
                main_results = results['summary'][original_files[0]]
                pass_at_1 = main_results.get('pass@1', 0) * 100
                
                print(f"\n🎯 Performance Analysis:")
                print(f"Current Pass@1: {pass_at_1:.1f}%")
                print(f"Expected GPT-4o: 50-60%")
                print(f"Expected GPT-4o-mini: 35-45%")
                
                if pass_at_1 < 35:
                    print(f"⚠️  Performance below expected range")
                    print(f"   Gap: {35 - pass_at_1:.1f}% below GPT-4o-mini minimum")
                elif pass_at_1 >= 35 and pass_at_1 <= 45:
                    print(f"✅ Performance within GPT-4o-mini expected range")
                elif pass_at_1 > 45:
                    print(f"🎉 Performance above GPT-4o-mini expected range")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="BigCodeBench-compatible Evaluator")
    parser.add_argument("--model", required=True,
                       help="Model name (e.g., 'o4-mini')")
    parser.add_argument("--execution", default="local",
                       choices=["local", "gradio", "e2b"],
                       help="Execution mode for evaluation")
    parser.add_argument("--file",
                       help="Evaluate only a specific file")
    parser.add_argument("--install-only", action="store_true",
                       help="Only install BigCodeBench, don't run evaluation")
    
    args = parser.parse_args()
    
    evaluator = BigCodeBenchEvaluator(args.model)
    
    if args.install_only:
        if evaluator.check_bigcodebench_installation():
            print("✅ BigCodeBench already installed")
        else:
            evaluator.install_bigcodebench()
        return
    
    if args.execution != "local":
        print("⚠️  Note: Remote execution (gradio/e2b) is recommended for safety")
        print("   Local execution runs untrusted code on your machine")
    
    # Run evaluation
    results = evaluator.evaluate_all_files(args.execution)
    
    # Analyze results
    evaluator.analyze_results(results)
    
    # Save results
    output_file = f"{args.model}_bigcodebench_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()