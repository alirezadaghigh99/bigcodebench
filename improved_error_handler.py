#!/usr/bin/env python3
"""
Improved error handling for BigCodeBench testing.
Provides better error categorization and recovery mechanisms.
"""

import ast
import sys
import traceback
import subprocess
from typing import Tuple, Dict, Optional
from pathlib import Path


class ImprovedErrorHandler:
    """Enhanced error handling for test execution."""
    
    def __init__(self):
        self.error_patterns = {
            'missing_module': [
                'ModuleNotFoundError',
                'ImportError',
                'No module named'
            ],
            'syntax_error': [
                'SyntaxError',
                'IndentationError',
                'TabError'
            ],
            'runtime_error': [
                'AttributeError',
                'TypeError',
                'ValueError',
                'KeyError',
                'IndexError',
                'NameError'
            ],
            'test_failure': [
                'AssertionError',
                'FAIL:',
                'FAILED'
            ],
            'timeout': [
                'TimeoutExpired',
                'timeout',
                'timed out'
            ]
        }
    
    def validate_code_before_execution(self, code: str) -> Tuple[bool, str]:
        """Validate code syntax and common issues before execution."""
        try:
            # Check basic syntax
            ast.parse(code)
            
            # Check for common problematic patterns
            issues = []
            
            # Check for undefined variables (basic check)
            if 'undefined_var' in code:
                issues.append("Potential undefined variable")
            
            # Check for missing imports
            lines = code.split('\n')
            import_lines = [line for line in lines if line.strip().startswith(('import ', 'from '))]
            function_lines = [line for line in lines if 'def ' in line]
            
            if len(function_lines) > 0 and len(import_lines) == 0:
                # Check if code uses common modules without importing
                common_modules = ['os', 'sys', 'json', 'math', 'random', 'datetime', 're']
                for module in common_modules:
                    if f'{module}.' in code and f'import {module}' not in code:
                        issues.append(f"Possible missing import: {module}")
            
            if issues:
                return True, f"Warning: {'; '.join(issues)}"
            
            return True, "Code validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e.msg} at line {e.lineno}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def attempt_code_repair(self, code: str, error_msg: str) -> Tuple[str, bool]:
        """Attempt to repair common code issues."""
        repaired_code = code
        repairs_made = []
        
        # Fix common indentation issues
        if 'IndentationError' in error_msg:
            lines = code.split('\n')
            fixed_lines = []
            for line in lines:
                # Fix mixed tabs and spaces
                if '\t' in line and '    ' in line:
                    # Convert tabs to spaces
                    line = line.expandtabs(4)
                    repairs_made.append("Fixed mixed tabs/spaces")
                fixed_lines.append(line)
            repaired_code = '\n'.join(fixed_lines)
        
        # Fix missing imports for common modules
        if 'ModuleNotFoundError' in error_msg and 'No module named' in error_msg:
            missing_module = None
            if "'numpy'" in error_msg:
                missing_module = "import numpy as np"
            elif "'pandas'" in error_msg:
                missing_module = "import pandas as pd"
            elif "'matplotlib'" in error_msg:
                missing_module = "import matplotlib.pyplot as plt"
            elif "'sklearn'" in error_msg:
                missing_module = "from sklearn import *"
            
            if missing_module and missing_module not in repaired_code:
                repaired_code = missing_module + "\n" + repaired_code
                repairs_made.append(f"Added missing import: {missing_module}")
        
        return repaired_code, len(repairs_made) > 0
    
    def install_missing_module_on_demand(self, module_name: str) -> bool:
        """Attempt to install missing modules on demand."""
        # Mapping of import names to pip package names
        module_mapping = {
            'cv2': 'opencv-python',
            'sklearn': 'scikit-learn',
            'PIL': 'Pillow',
            'bs4': 'beautifulsoup4',
        }
        
        pip_name = module_mapping.get(module_name, module_name)
        
        try:
            print(f"  🔄 Attempting to install {pip_name}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", pip_name
            ], check=True, capture_output=True, text=True, timeout=60)
            
            print(f"  ✅ Successfully installed {pip_name}")
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  ❌ Failed to install {pip_name}: {e}")
            return False
    
    def categorize_error(self, stderr: str) -> Dict[str, str]:
        """Categorize error with enhanced detail."""
        stderr_lower = stderr.lower()
        
        category = "unknown"
        details = stderr.strip()
        suggestion = "No specific suggestion available"
        
        for error_type, patterns in self.error_patterns.items():
            if any(pattern.lower() in stderr_lower for pattern in patterns):
                category = error_type
                break
        
        # Provide specific suggestions based on category
        if category == "missing_module":
            if "no module named '" in stderr_lower:
                module = stderr.split("no module named '")[1].split("'")[0]
                suggestion = f"Install missing module: pip install {module}"
        
        elif category == "syntax_error":
            suggestion = "Check code syntax, indentation, and brackets"
        
        elif category == "runtime_error":
            if "attributeerror" in stderr_lower:
                suggestion = "Check object attributes and method names"
            elif "typeerror" in stderr_lower:
                suggestion = "Check function arguments and data types"
            elif "valueerror" in stderr_lower:
                suggestion = "Check input values and ranges"
        
        elif category == "test_failure":
            suggestion = "Generated code logic doesn't match expected behavior"
        
        elif category == "timeout":
            suggestion = "Code execution took too long, check for infinite loops"
        
        return {
            "category": category,
            "details": details[:500],  # Limit details length
            "suggestion": suggestion
        }


# Make the error handler available for import
error_handler = ImprovedErrorHandler()
