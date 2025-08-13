#!/usr/bin/env python3
"""
Standardize function names in BigCodeBench generated code files.
Changes all function definitions to use 'task_func' instead of various names.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any


def extract_function_name(code: str) -> str:
    """Extract the function name from Python code."""
    pattern = r'^def\s+(\w+)\s*\('
    match = re.search(pattern, code, re.MULTILINE)
    return match.group(1) if match else None


def standardize_function_name(code: str, target_name: str = "task_func") -> str:
    """
    Replace function definition name with target_name in Python code.
    
    Args:
        code: Python code string
        target_name: New function name to use (default: "task_func")
    
    Returns:
        Modified code with standardized function name
    """
    if not code or not code.strip():
        return code
    
    # Find the function definition
    func_def_pattern = r'^(def\s+)(\w+)(\s*\()'
    match = re.search(func_def_pattern, code, re.MULTILINE)
    
    if not match:
        return code
    
    original_name = match.group(2)
    
    # Replace function definition
    new_code = re.sub(
        func_def_pattern,
        rf'\1{target_name}\3',
        code,
        count=1,
        flags=re.MULTILINE
    )
    
    # Replace any calls to the original function with the new name
    # Only replace standalone function calls, not method calls or partial matches
    call_pattern = rf'\b{re.escape(original_name)}\b(?=\s*\()'
    new_code = re.sub(call_pattern, target_name, new_code)
    
    return new_code


def process_jsonl_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    """
    Process a JSONL file and standardize function names.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total_lines": 0,
        "modified_lines": 0,
        "errors": 0,
        "function_names_found": set()
    }
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            stats["total_lines"] += 1
            
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Extract and modify response_code if it exists
                if "response_code" in data and data["response_code"]:
                    original_code = data["response_code"]
                    original_func_name = extract_function_name(original_code)
                    
                    if original_func_name:
                        stats["function_names_found"].add(original_func_name)
                        
                        # Standardize function name
                        modified_code = standardize_function_name(original_code, "task_func")
                        
                        if modified_code != original_code:
                            data["response_code"] = modified_code
                            stats["modified_lines"] += 1
                            print(f"  Line {line_num}: {original_func_name} → task_func")
                
                # Also modify the response field if it contains code
                if "response" in data and data["response"]:
                    response = data["response"]
                    
                    # Extract code blocks from markdown
                    code_block_pattern = r'```python\n(.*?)```'
                    matches = re.findall(code_block_pattern, response, re.DOTALL)
                    
                    for code_block in matches:
                        original_func_name = extract_function_name(code_block)
                        if original_func_name:
                            modified_code = standardize_function_name(code_block, "task_func")
                            if modified_code != code_block:
                                response = response.replace(code_block, modified_code)
                    
                    data["response"] = response
                
                # Write modified JSON line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"  Error parsing line {line_num}: {e}")
                stats["errors"] += 1
                # Write original line if there's an error
                outfile.write(line)
            except Exception as e:
                print(f"  Unexpected error on line {line_num}: {e}")
                stats["errors"] += 1
                outfile.write(line)
    
    return stats


def main():
    """Main function to process all code generation files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize function names in generated code")
    parser.add_argument("--input-dir", default="generation_output/o4",
                       help="Input directory containing code_*.jsonl files")
    parser.add_argument("--output-dir", default="generation_output/o4_standardized",
                       help="Output directory for standardized files")
    
    args = parser.parse_args()
    
    # Define paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all code generation files
    code_files = list(input_dir.glob("code_*.jsonl"))
    
    if not code_files:
        print("No code generation files found in", input_dir)
        return
    
    print(f"Found {len(code_files)} code generation files")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    total_stats = {
        "files_processed": 0,
        "total_lines": 0,
        "modified_lines": 0,
        "errors": 0,
        "all_function_names": set()
    }
    
    # Process each file
    for input_file in sorted(code_files):
        print(f"Processing: {input_file.name}")
        
        output_file = output_dir / input_file.name
        stats = process_jsonl_file(input_file, output_file)
        
        # Update total stats
        total_stats["files_processed"] += 1
        total_stats["total_lines"] += stats["total_lines"]
        total_stats["modified_lines"] += stats["modified_lines"]
        total_stats["errors"] += stats["errors"]
        total_stats["all_function_names"].update(stats["function_names_found"])
        
        print(f"  Lines processed: {stats['total_lines']}")
        print(f"  Lines modified: {stats['modified_lines']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Function names found: {sorted(stats['function_names_found'])}")
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total lines: {total_stats['total_lines']}")
    print(f"Lines modified: {total_stats['modified_lines']}")
    print(f"Errors: {total_stats['errors']}")
    print(f"Modification rate: {total_stats['modified_lines']/total_stats['total_lines']*100:.1f}%")
    print()
    print("All function names found:")
    for name in sorted(total_stats['all_function_names']):
        print(f"  - {name}")
    
    print(f"\nStandardized files saved to: {output_dir}")


if __name__ == "__main__":
    main()