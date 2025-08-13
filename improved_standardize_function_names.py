#!/usr/bin/env python3
"""
Improved standardize function names in BigCodeBench generated code files.
Changes all function definitions to use 'task_func' instead of various names.
Includes better error handling and validation.
"""

import json
import os
import re
import ast
from pathlib import Path
from typing import Dict, Any, List, Tuple
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def extract_function_name(code: str) -> str:
    """Extract the function name from Python code using AST parsing."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        # Fallback to regex if AST parsing fails
        pattern = r'^def\s+(\w+)\s*\('
        match = re.search(pattern, code, re.MULTILINE)
        return match.group(1) if match else None
    return None


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python syntax and return error message if invalid."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def standardize_function_name(code: str, target_name: str = "task_func") -> Tuple[str, bool, str]:
    """
    Replace function definition name with target_name in Python code.
    
    Args:
        code: Python code string
        target_name: New function name to use (default: "task_func")
    
    Returns:
        Tuple of (modified_code, success, error_message)
    """
    if not code or not code.strip():
        return code, True, ""
    
    # First validate original syntax
    valid, error = validate_python_syntax(code)
    if not valid:
        return code, False, f"Original code invalid: {error}"
    
    # Find the function definition using AST
    try:
        tree = ast.parse(code)
        original_name = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                original_name = node.name
                break
        
        if not original_name:
            return code, True, "No function definition found"
        
        if original_name == target_name:
            return code, True, "Function name already correct"
        
        # Replace function definition more carefully
        func_def_pattern = rf'^(\s*def\s+){re.escape(original_name)}(\s*\()'
        new_code = re.sub(
            func_def_pattern,
            rf'\1{target_name}\2',
            code,
            count=1,
            flags=re.MULTILINE
        )
        
        # Replace function calls more carefully - only standalone calls
        # Use word boundaries and look ahead to ensure we're matching a function call
        call_pattern = rf'\b{re.escape(original_name)}(?=\s*\()'
        new_code = re.sub(call_pattern, target_name, new_code)
        
        # Validate the modified code
        valid, error = validate_python_syntax(new_code)
        if not valid:
            return code, False, f"Modified code invalid: {error}"
        
        return new_code, True, ""
        
    except Exception as e:
        return code, False, f"Error during standardization: {str(e)}"


def process_jsonl_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
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
        "syntax_errors": 0,
        "function_names_found": set(),
        "error_details": []
    }
    
    # Count total lines first for progress bar
    total_lines = 0
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.strip():
                total_lines += 1
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        # Initialize progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=total_lines,
                desc=f"Processing {input_path.name}",
                unit="lines",
                ncols=100
            )
        
        for line_num, line in enumerate(infile, 1):
            stats["total_lines"] += 1
            
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Process response_code if it exists
                if "response_code" in data and data["response_code"]:
                    original_code = data["response_code"]
                    original_func_name = extract_function_name(original_code)
                    
                    if original_func_name:
                        stats["function_names_found"].add(original_func_name)
                        
                        # Standardize function name
                        modified_code, success, error_msg = standardize_function_name(original_code, "task_func")
                        
                        if success:
                            if modified_code != original_code:
                                data["response_code"] = modified_code
                                stats["modified_lines"] += 1
                                if TQDM_AVAILABLE:
                                    pbar.set_postfix_str(f"✅ {original_func_name} → task_func")
                                else:
                                    print(f"  Line {line_num}: {original_func_name} → task_func")
                        else:
                            stats["errors"] += 1
                            if "syntax" in error_msg.lower():
                                stats["syntax_errors"] += 1
                            stats["error_details"].append(f"Line {line_num}: {error_msg}")
                            if TQDM_AVAILABLE:
                                pbar.set_postfix_str(f"❌ Error: {error_msg[:30]}...")
                            else:
                                print(f"  Error on line {line_num}: {error_msg}")
                
                # Also process the response field if it contains code blocks
                if "response" in data and data["response"]:
                    response = data["response"]
                    
                    # Extract code blocks from markdown
                    code_block_pattern = r'```python\n(.*?)```'
                    modified_response = response
                    
                    for match in re.finditer(code_block_pattern, response, re.DOTALL):
                        code_block = match.group(1)
                        original_func_name = extract_function_name(code_block)
                        
                        if original_func_name:
                            modified_code, success, error_msg = standardize_function_name(code_block, "task_func")
                            if success and modified_code != code_block:
                                modified_response = modified_response.replace(code_block, modified_code)
                    
                    data["response"] = modified_response
                
                # Write modified JSON line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                if TQDM_AVAILABLE:
                    pbar.update(1)
                
            except json.JSONDecodeError as e:
                if not TQDM_AVAILABLE:
                    print(f"  JSON parse error on line {line_num}: {e}")
                stats["errors"] += 1
                stats["error_details"].append(f"Line {line_num}: JSON parse error - {e}")
                # Write original line if there's an error
                outfile.write(line)
                if TQDM_AVAILABLE:
                    pbar.update(1)
                    pbar.set_postfix_str(f"⚠️  JSON error")
            except Exception as e:
                if not TQDM_AVAILABLE:
                    print(f"  Unexpected error on line {line_num}: {e}")
                stats["errors"] += 1
                stats["error_details"].append(f"Line {line_num}: Unexpected error - {e}")
                outfile.write(line)
                if TQDM_AVAILABLE:
                    pbar.update(1)
                    pbar.set_postfix_str(f"💥 Error")
        
        # Close progress bar
        if TQDM_AVAILABLE:
            pbar.close()
    
    return stats


def main():
    """Main function to process all code generation files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved standardize function names in generated code")
    parser.add_argument("--input-dir", default="generation_output/o4",
                       help="Input directory containing code_*.jsonl files")
    parser.add_argument("--output-dir", default="generation_output/o4_standardized",
                       help="Output directory for standardized files")
    parser.add_argument("--validate", action="store_true",
                       help="Validate syntax of all processed files")
    
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
        "syntax_errors": 0,
        "all_function_names": set(),
        "all_error_details": []
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
        total_stats["syntax_errors"] += stats["syntax_errors"]
        total_stats["all_function_names"].update(stats["function_names_found"])
        total_stats["all_error_details"].extend(stats["error_details"])
        
        print(f"  Lines processed: {stats['total_lines']}")
        print(f"  Lines modified: {stats['modified_lines']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Syntax errors: {stats['syntax_errors']}")
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
    print(f"Syntax errors: {total_stats['syntax_errors']}")
    print(f"Modification rate: {total_stats['modified_lines']/total_stats['total_lines']*100:.1f}%")
    print(f"Error rate: {total_stats['errors']/total_stats['total_lines']*100:.1f}%")
    print(f"Syntax error rate: {total_stats['syntax_errors']/total_stats['total_lines']*100:.1f}%")
    print()
    print("All function names found:")
    for name in sorted(total_stats['all_function_names']):
        print(f"  - {name}")
    
    if total_stats['all_error_details']:
        print("\nError details:")
        for detail in total_stats['all_error_details'][:20]:  # Show first 20 errors
            print(f"  {detail}")
        if len(total_stats['all_error_details']) > 20:
            print(f"  ... and {len(total_stats['all_error_details']) - 20} more errors")
    
    print(f"\nStandardized files saved to: {output_dir}")


if __name__ == "__main__":
    main()