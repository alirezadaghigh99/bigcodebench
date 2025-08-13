#!/usr/bin/env python3
"""
Debug script to understand why function definitions aren't being removed.
"""

test_code = '''import subprocess
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
        self.assertEqual(result, ['file1.txt', 'file2.img'])'''

def debug_remove_function_definitions(test_code: str) -> str:
    """Debug version of remove_function_definitions."""
    lines = test_code.split('\n')
    cleaned_lines = []
    in_function_def = False
    function_indent = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        current_indent = len(line) - len(line.lstrip())
        
        print(f"Line {i+1}: indent={current_indent}, in_func={in_function_def}, stripped='{line_stripped}'")
        
        # Check if this is a function definition (but not a test method or class method)
        if line_stripped.startswith('def ') and not line_stripped.startswith('def test_'):
            # Extract function name more carefully
            function_name = line_stripped.split('(')[0].replace('def ', '').strip()
            
            print(f"  Found function definition: '{function_name}'")
            
            # These are functions under test that should be removed
            # Be more aggressive about removing any standalone function definitions
            if function_name in ['task_func', 'test_func', 'func', 'solution', 'my_function', 'your_function'] or \
               any(name in function_name.lower() for name in ['task_func', 'test_func', 'solution', 'my_function', 'your_function']) or \
               current_indent == 0:  # Top-level function definitions should be removed
                print(f"  -> REMOVING function '{function_name}' (indent={current_indent})")
                in_function_def = True
                function_indent = current_indent
                continue
            else:
                print(f"  -> KEEPING function '{function_name}' (indent={current_indent})")
        
        # If we're inside a function definition, skip lines until we're back to the original indent level or less
        if in_function_def:
            if line_stripped == '':  # Empty line
                print(f"  -> SKIPPING empty line")
                continue
            elif current_indent > function_indent:  # Still inside function
                print(f"  -> SKIPPING line inside function (indent {current_indent} > {function_indent})")
                continue
            else:  # Back to original indent level or less, function definition ended
                print(f"  -> ENDING function definition (indent {current_indent} <= {function_indent})")
                in_function_def = False
                function_indent = 0
                # Don't continue here, we want to process this line normally
        
        # Keep the line if we're not in a function definition
        if not in_function_def:
            print(f"  -> KEEPING line")
            cleaned_lines.append(line)
        else:
            print(f"  -> SKIPPING line (in function)")
    
    return '\n'.join(cleaned_lines)

result = debug_remove_function_definitions(test_code)
print("\n=== RESULT ===")
print(result)