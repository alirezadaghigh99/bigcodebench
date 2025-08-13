#!/usr/bin/env python3
"""
Test script to verify the improved test cleaning functions.
"""

import re
import ast
from typing import Optional


def extract_function_name(code: str) -> Optional[str]:
    """Extract the main function name from Python code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        # Fallback regex method
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
    return None


def convert_function_to_task_func(code: str) -> str:
    """Convert the main function name to task_func."""
    original_name = extract_function_name(code)
    if not original_name:
        return code
    
    # Replace function definition
    pattern = rf'\bdef\s+{re.escape(original_name)}\s*\('
    replacement = 'def task_func('
    converted_code = re.sub(pattern, replacement, code)
    
    return converted_code


def remove_function_definitions(test_code: str) -> str:
    """Remove function definitions from test code, keeping only imports and test classes."""
    lines = test_code.split('\n')
    cleaned_lines = []
    in_function_def = False
    function_indent = 0
    
    for line in lines:
        line_stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())
        
        # Check if this is a function definition
        if line_stripped.startswith('def '):
            # Extract function name more carefully
            function_name = line_stripped.split('(')[0].replace('def ', '').strip()
            
            # Skip unittest test methods (they start with 'test_' and are indented)
            if function_name.startswith('test_') and current_indent > 0:
                # This is a unittest test method, keep it
                pass
            else:
                # These are functions under test that should be removed
                # Be more aggressive about removing any standalone function definitions
                if function_name in ['task_func', 'test_func', 'func', 'solution', 'my_function', 'your_function'] or \
                   any(name in function_name.lower() for name in ['task_func', 'test_func', 'solution', 'my_function', 'your_function']) or \
                   current_indent == 0:  # Top-level function definitions should be removed
                    in_function_def = True
                    function_indent = current_indent
                    continue
        
        # If we're inside a function definition, skip lines until we're back to the original indent level or less
        if in_function_def:
            if line_stripped == '':  # Empty line
                continue
            elif current_indent > function_indent:  # Still inside function
                continue
            else:  # Back to original indent level or less, function definition ended
                in_function_def = False
                function_indent = 0
                # Don't continue here, we want to process this line normally
        
        # Keep the line if we're not in a function definition
        if not in_function_def:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_test_imports(test_code: str) -> str:
    """Clean problematic imports from test code."""
    lines = test_code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip problematic imports
        if (line_stripped.startswith('import ') or line_stripped.startswith('from ')) and \
           any(x in line_stripped for x in [
               'your_module', 'my_module', 'task', 'solution', 'module', 
               'task_module', 'your_solution', 'my_solution', 'test_module',
               'from task ', 'import task', 'task import', 'solution import',
               'your_', 'my_', 'module_', 'task_', 'solution_',
               'import your', 'import my', 'import task', 'import solution',
               'from your', 'from my', 'from task', 'from solution'
           ]):
            # Skip this line completely
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def convert_test_calls_to_task_func(test_code: str) -> str:
    """Convert function calls in test code to use task_func."""
    # Define built-in functions that should NOT be replaced
    builtin_functions = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
        'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
        'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
        'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
        'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
        'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
        'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
        'super', 'tuple', 'type', 'vars', 'zip', '__import__',
        # Common test methods that should not be replaced
        'setUp', 'tearDown', 'assertEqual', 'assertTrue', 'assertFalse', 
        'assertRaises', 'assertIn', 'assertNotIn', 'mock', 'patch', 'call'
    }
    
    # Replace various function call patterns with task_func
    patterns = [
        (r'\btest_func\s*\(', 'task_func('),
        (r'\byour_function\s*\(', 'task_func('),
        (r'\bmy_function\s*\(', 'task_func('),
        (r'\bsolution\s*\(', 'task_func('),
        (r'\btask\s*\(', 'task_func('),
        (r'\bfunc\s*\(', 'task_func('),  # This is important - "func" is often the function under test
    ]
    
    for pattern, replacement in patterns:
        # First, find all matches
        matches = re.finditer(pattern, test_code)
        
        # Process matches in reverse order to avoid index issues when replacing
        for match in reversed(list(matches)):
            # Extract the function name being called
            func_name = match.group(0).replace('(', '').strip()
            
            # Skip if it's a built-in function or common test method
            if func_name not in builtin_functions:
                # Check if this is not a method call (no dot before it)
                start_pos = match.start()
                if start_pos == 0 or test_code[start_pos - 1] not in '.':
                    test_code = test_code[:match.start()] + replacement + test_code[match.end():]
    
    return test_code


def test_cleaning():
    """Test the cleaning functions with sample data."""
    
    # Sample test code with function definitions that should be removed
    sample_test = '''import subprocess
import ftplib
import os
import unittest
from unittest.mock import patch, MagicMock, call

# The function to be tested (as specified in the prompt)
# Per instructions, the function is named test_func and is not implemented here.
# The test suite below assumes a correct implementation will be provided at runtime.
def test_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    pass

class TestFtpDownload(unittest.TestCase):

    @patch('__main__.subprocess.run')
    @patch('__main__.ftplib.FTP')
    def test_successful_download(self, mock_ftp_class, mock_subprocess_run):
        """
        Tests the happy path: successful connection, login, and download of multiple files.
        """
        # Arrange
        mock_ftp_instance = mock_ftp_class.return_value
        mock_ftp_instance.nlst.return_value = ['file1.txt', 'file2.img']
        
        # Act
        result = test_func()

        # Assert
        self.assertEqual(result, ['file1.txt', 'file2.img'])
        mock_ftp_class.assert_called_once_with('ftp.dlptest.com')
        mock_ftp_instance.login.assert_called_once_with(user='dlpuser', passwd='rNrKYTX9g7z3RgJRmxWuGHbeu')
        mock_ftp_instance.cwd.assert_called_once_with('/ftp/test')
        mock_ftp_instance.nlst.assert_called_once()
        mock_ftp_instance.quit.assert_called_once()
        
        expected_calls = [
            call(['wget', 'ftp://dlpuser:rNrKYTX9g7z3RgJRmxWuGHbeu@ftp.dlptest.com/ftp/test/file1.txt']),
            call(['wget', 'ftp://dlpuser:rNrKYTX9g7z3RgJRmxWuGHbeu@ftp.dlptest.com/ftp/test/file2.img'])
        ]
        mock_subprocess_run.assert_has_calls(expected_calls, any_order=True)'''
    
    print("=== ORIGINAL TEST CODE ===")
    print(sample_test)
    
    # Step 1: Remove function definitions
    print("\n=== AFTER REMOVING FUNCTION DEFINITIONS ===")
    step1 = remove_function_definitions(sample_test)
    print(step1)
    
    # Step 2: Clean imports
    print("\n=== AFTER CLEANING IMPORTS ===")
    step2 = clean_test_imports(step1)
    print(step2)
    
    # Step 3: Convert function calls
    print("\n=== AFTER CONVERTING FUNCTION CALLS ===")
    step3 = convert_test_calls_to_task_func(step2)
    print(step3)
    
    # Test code generation
    sample_code = '''def download_files(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):
    import subprocess
    import ftplib
    import os
    
    try:
        ftp = ftplib.FTP(ftp_server)
        ftp.login(user=ftp_user, passwd=ftp_password)
        ftp.cwd(ftp_dir)
        
        files = ftp.nlst()
        
        for file in files:
            url = f"ftp://{ftp_user}:{ftp_password}@{ftp_server}{ftp_dir}/{file}"
            subprocess.run(['wget', url])
        
        ftp.quit()
        return files
    except Exception as e:
        raise Exception(f"Failed to connect to FTP server {ftp_server}: {str(e)}")'''
    
    print("\n=== ORIGINAL CODE ===")
    print(sample_code)
    
    print("\n=== AFTER CONVERTING TO task_func ===")
    converted_code = convert_function_to_task_func(sample_code)
    print(converted_code)


if __name__ == "__main__":
    test_cleaning()