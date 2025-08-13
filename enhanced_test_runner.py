#!/usr/bin/env python3
"""
Enhanced test runner with improved error handling and recovery mechanisms.
"""

import json
import os
import sys
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from improved_error_handler import error_handler
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False
    print("Warning: Enhanced error handler not available")


class EnhancedTestRunner:
    """Enhanced test runner with better error handling."""
    
    def __init__(self, dataset_path: str, generated_code_dir: str, max_workers: int = None):
        self.dataset_path = Path(dataset_path)
        self.generated_code_dir = Path(generated_code_dir)
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.print_lock = Lock()
        self.setup_lock = Lock()
        self.shared_env_dir = None
        self.python_exe = None
        self.auto_install_modules = True  # Enable automatic module installation
        self.installed_modules = set()  # Track already installed modules
    
    def test_with_recovery(self, task_id: str, task_data: Dict, generated_code: str, max_retries: int = 2) -> Dict:
        """Test with error recovery and retry mechanisms."""
        result = {
            'id': task_id,
            'test_result': 0,
            'failure_reason': None,
            'debug_info': {},
            'retry_count': 0,
            'recovery_attempts': []
        }
        
        original_code = generated_code
        
        for attempt in range(max_retries + 1):
            result['retry_count'] = attempt
            
            # Validate code before execution if error handler is available
            if ERROR_HANDLER_AVAILABLE:
                valid, validation_msg = error_handler.validate_code_before_execution(generated_code)
                if not valid:
                    result['recovery_attempts'].append(f"Attempt {attempt}: Validation failed - {validation_msg}")
                    
                    # Try to repair the code
                    repaired_code, was_repaired = error_handler.attempt_code_repair(generated_code, validation_msg)
                    if was_repaired:
                        generated_code = repaired_code
                        result['recovery_attempts'].append(f"Attempt {attempt}: Code repair attempted")
                        continue
                    else:
                        result['failure_reason'] = f"Code validation failed: {validation_msg}"
                        break
            
            # Run the test
            test_result = self._run_single_test(task_id, task_data, generated_code)
            
            if test_result['test_result'] == 1:
                # Success!
                result.update(test_result)
                break
            
            # Test failed, analyze the error
            failure_reason = test_result.get('failure_reason', '')
            
            # Try recovery if error handler is available
            if ERROR_HANDLER_AVAILABLE and attempt < max_retries:
                error_info = error_handler.categorize_error(test_result.get('debug_info', {}).get('stderr', ''))
                result['recovery_attempts'].append(f"Attempt {attempt}: {error_info['category']} - {error_info['suggestion']}")
                
                # Try to handle missing modules
                if error_info['category'] == 'missing_module' and self.auto_install_modules:
                    stderr = test_result.get('debug_info', {}).get('stderr', '')
                    if "no module named '" in stderr.lower():
                        module_name = stderr.split("no module named '")[1].split("'")[0]
                        if module_name not in self.installed_modules:
                            if error_handler.install_missing_module_on_demand(module_name):
                                self.installed_modules.add(module_name)
                                result['recovery_attempts'].append(f"Attempt {attempt}: Installed {module_name}")
                                continue
                
                # Try code repair for syntax/runtime errors
                if error_info['category'] in ['syntax_error', 'runtime_error']:
                    repaired_code, was_repaired = error_handler.attempt_code_repair(generated_code, failure_reason)
                    if was_repaired and repaired_code != generated_code:
                        generated_code = repaired_code
                        result['recovery_attempts'].append(f"Attempt {attempt}: Code repair for {error_info['category']}")
                        continue
            
            # If this is the last attempt or no recovery possible, use this result
            if attempt == max_retries:
                result.update(test_result)
        
        return result
    
    def _run_single_test(self, task_id: str, task_data: Dict, generated_code: str) -> Dict:
        """Run a single test (original implementation)."""
        # This would be the original test_single_task implementation
        # For now, return a simplified version
        return {
            'id': task_id,
            'test_result': 0,
            'failure_reason': "Placeholder - implement actual test execution",
            'debug_info': {}
        }


def main():
    """Main function for enhanced testing."""
    print("🚀 Enhanced BigCodeBench Test Runner")
    print("Features: Error recovery, auto-module installation, code repair")
    
    # This would integrate with the existing pipeline
    pass


if __name__ == "__main__":
    main()
