#!/usr/bin/env python3
"""
Debug script to understand the exact string issue.
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

lines = test_code.split('\n')
for i, line in enumerate(lines):
    if 'def test_func' in line:
        print(f"Line {i+1}: '{line}'")
        print(f"  Repr: {repr(line)}")
        print(f"  Stripped: '{line.strip()}'")
        print(f"  Starts with 'def ': {line.strip().startswith('def ')}")
        print(f"  Starts with 'def test_': {line.strip().startswith('def test_')}")
        print()
        
        # Let's try to split it
        try:
            function_name = line.strip().split('(')[0].replace('def ', '').strip()
            print(f"  Function name: '{function_name}'")
            print(f"  Function name repr: {repr(function_name)}")
            print(f"  Is 'test_func': {function_name == 'test_func'}")
            print(f"  In list: {function_name in ['task_func', 'test_func', 'func', 'solution', 'my_function', 'your_function']}")
        except Exception as e:
            print(f"  Error parsing: {e}")