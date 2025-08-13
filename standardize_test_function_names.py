#!/usr/bin/env python3
"""
Standardize test files to use test_func instead of func.
This script processes test_*.jsonl files in the generation output directory and:
1. Changes function calls from func(...) to test_func(...)
2. Removes import statements like 'from your_module import func'
3. Removes import statements like 'from solution import func'
4. Replaces module-specific imports with generic test_func references
5. Only changes user-defined function calls, not built-in or third-party functions
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class TestFunctionStandardizer:
    """Standardizes test function names in generated test files."""
    
    def __init__(self):
        # Built-in and common third-party functions that should NOT be changed
        self.builtin_functions = {
            # Python builtins
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
            'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
            'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
            'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
            'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
            'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
            'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
            'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
            'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
            'super', 'tuple', 'type', 'vars', 'zip',
            
            # Common testing functions
            'setUp', 'tearDown', 'assertEqual', 'assertTrue', 'assertFalse',
            'assertIn', 'assertNotIn', 'assertIsNone', 'assertIsNotNone',
            'assertRaises', 'assertCountEqual', 'assertGreater', 'assertLess',
            'assertIsInstance', 'mock', 'patch', 'call', 'MagicMock',
            
            # Common third-party functions
            'join', 'split', 'strip', 'replace', 'find', 'index', 'append',
            'remove', 'pop', 'insert', 'extend', 'clear', 'copy', 'count',
            'reverse', 'sort', 'keys', 'values', 'items', 'get', 'update',
            'close', 'read', 'write', 'readline', 'readlines', 'flush',
            'seek', 'tell', 'exists', 'isfile', 'isdir', 'makedirs', 'listdir',
            'remove', 'rename', 'rmdir', 'getcwd', 'chdir', 'path', 'basename',
            'dirname', 'splitext', 'abspath', 'realpath', 'normpath', 'relpath',
            'connect', 'login', 'quit', 'nlst', 'cwd', 'run', 'check', 'call',
            'Popen', 'PIPE', 'communicate', 'wait', 'poll', 'terminate', 'kill',
            'process_iter', 'wait_procs', 'sleep'
        }
        
        # Patterns for imports to remove
        self.import_patterns = [
            # from your_module import func
            r'from\s+your_module\s+import\s+func\s*',
            # from solution import func  
            r'from\s+solution\s+import\s+func\s*',
            # from module_name import func
            r'from\s+\w+\s+import\s+func\s*',
            # import your_module
            r'import\s+your_module\s*',
            # import solution
            r'import\s+solution\s*'
        ]
    
    def should_replace_function(self, func_name: str, context: str) -> bool:
        """
        Determine if a function call should be replaced with test_func.
        Only replace 'func' calls that appear to be the main function under test.
        """
        # Don't replace if it's a builtin or common function
        if func_name in self.builtin_functions:
            return False
            
        # Only replace 'func' (the main function under test)
        if func_name != 'func':
            return False
            
        # Don't replace if it's a method call (has a dot before it)
        if re.search(r'\w+\.\s*func\s*\(', context):
            return False
            
        # Don't replace if it's being defined (def func)
        if re.search(r'def\s+func\s*\(', context):
            return False
            
        # Don't replace if it's in a comment
        if '#' in context.split('func')[0].split('\n')[-1]:
            return False
            
        return True
    
    def standardize_code_content(self, code: str) -> str:
        """Standardize the code content by replacing function calls and removing imports."""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # Skip lines that match import patterns
            skip_line = False
            for pattern in self.import_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    skip_line = True
                    break
            
            if skip_line:
                continue
            
            # Replace func(...) with test_func(...) in function calls
            # Use word boundaries to avoid replacing parts of other words
            modified_line = line
            
            # Find all 'func' occurrences and check if they should be replaced
            for match in re.finditer(r'\bfunc\b', line):
                start, end = match.span()
                # Get context around the match
                context = line[max(0, start-20):min(len(line), end+20)]
                
                if self.should_replace_function('func', context):
                    # Replace this specific occurrence
                    modified_line = (modified_line[:start] + 
                                   'test_func' + 
                                   modified_line[end:])
                    # Adjust for the length difference
                    diff = len('test_func') - len('func')
                    # Update positions for subsequent matches
                    break  # Process one at a time to avoid position issues
            
            result_lines.append(modified_line)
        
        return '\n'.join(result_lines)
    
    def process_jsonl_file(self, file_path: Path) -> Tuple[int, int]:
        """
        Process a single JSONL file and standardize function names.
        Returns (total_processed, total_modified).
        """
        print(f"Processing {file_path.name}...")
        
        processed_count = 0
        modified_count = 0
        
        # Read all entries
        entries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
                        continue
        
        # Process each entry
        for entry in entries:
            processed_count += 1
            original_response_code = entry.get('response_code', '')
            
            if original_response_code:
                # Standardize the response_code
                standardized_code = self.standardize_code_content(original_response_code)
                
                if standardized_code != original_response_code:
                    entry['response_code'] = standardized_code
                    modified_count += 1
                    print(f"  Modified task {entry.get('task_id', 'unknown')}")
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return processed_count, modified_count
    
    def standardize_directory(self, directory_path: Path) -> None:
        """Standardize all test_*.jsonl files in a directory."""
        print(f"Standardizing test files in: {directory_path}")
        
        # Find all test_*.jsonl files
        test_files = list(directory_path.glob("test_*.jsonl"))
        
        if not test_files:
            print("No test_*.jsonl files found!")
            return
        
        print(f"Found {len(test_files)} test files to process")
        
        total_processed = 0
        total_modified = 0
        
        for file_path in sorted(test_files):
            processed, modified = self.process_jsonl_file(file_path)
            total_processed += processed
            total_modified += modified
        
        print(f"\nSummary:")
        print(f"Files processed: {len(test_files)}")
        print(f"Total entries: {total_processed}")
        print(f"Entries modified: {total_modified}")
        
        if total_modified > 0:
            print(f"\n✅ Successfully standardized {total_modified} test entries!")
        else:
            print(f"\n✅ All test entries were already standardized!")


def main():
    """Main function to run the standardization process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize test function names in generated test files")
    parser.add_argument("--directory", "-d", 
                       default="generation_output/o4",
                       help="Directory containing test_*.jsonl files to standardize")
    parser.add_argument("--output-dir", "-o",
                       help="Output directory (default: create _standardized version)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.directory)
    
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist!")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create _standardized version
        output_dir = input_dir.parent / f"{input_dir.name}_test_standardized"
    
    # Create output directory and copy files
    if output_dir != input_dir:
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(exist_ok=True)
        
        # Copy all test_*.jsonl files to output directory
        test_files = list(input_dir.glob("test_*.jsonl"))
        for file_path in test_files:
            output_file = output_dir / file_path.name
            with open(file_path, 'r', encoding='utf-8') as src, \
                 open(output_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        
        print(f"Copied {len(test_files)} test files to {output_dir}")
        work_dir = output_dir
    else:
        work_dir = input_dir
    
    # Initialize standardizer and process files
    standardizer = TestFunctionStandardizer()
    standardizer.standardize_directory(work_dir)


if __name__ == "__main__":
    main()