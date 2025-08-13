#!/usr/bin/env python3
"""
Utility script to analyze detailed failure information from test results.
Provides insights into common failure patterns and specific error details.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_test_results(file_path: Path) -> List[Dict]:
    """Load test results from JSONL file."""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results


def load_debug_results(file_path: Path) -> List[Dict]:
    """Load debug results from JSON file."""
    if not file_path.exists():
        return []
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_error_patterns(results: List[Dict]) -> Dict[str, Any]:
    """Analyze error patterns and provide detailed breakdown."""
    analysis = {
        "total_tests": len(results),
        "passed_tests": 0,
        "failed_tests": 0,
        "error_categories": defaultdict(list),
        "specific_errors": defaultdict(int),
        "line_number_patterns": defaultdict(int),
        "module_issues": defaultdict(int)
    }
    
    for result in results:
        if result['test_result'] == 1:
            analysis["passed_tests"] += 1
        else:
            analysis["failed_tests"] += 1
            failure_reason = result.get('failure_reason', 'Unknown')
            
            # Categorize by main error type
            if 'Syntax error' in failure_reason:
                category = 'Syntax Error'
            elif 'Attribute error' in failure_reason:
                category = 'Attribute Error'
            elif 'Type error' in failure_reason:
                category = 'Type Error'
            elif 'Name error' in failure_reason:
                category = 'Name Error'
            elif 'Missing module' in failure_reason:
                category = 'Missing Module'
                # Extract module name
                if "install with: pip install" in failure_reason:
                    module = failure_reason.split("install with: pip install ")[1].split()[0]
                    analysis["module_issues"][module] += 1
            elif 'Test failure' in failure_reason:
                category = 'Test Assertion Failure'
            elif 'Test error' in failure_reason:
                category = 'Test Execution Error'
            elif 'Value error' in failure_reason:
                category = 'Value Error'
            elif 'Key error' in failure_reason:
                category = 'Key Error'
            elif 'Index error' in failure_reason:
                category = 'Index Error'
            elif 'File not found' in failure_reason:
                category = 'File Not Found'
            elif 'Permission error' in failure_reason:
                category = 'Permission Error'
            elif 'Test timeout' in failure_reason:
                category = 'Timeout'
            else:
                category = 'Other'
            
            analysis["error_categories"][category].append({
                "task_id": result['id'],
                "failure_reason": failure_reason
            })
            
            # Count specific error messages
            analysis["specific_errors"][failure_reason] += 1
            
            # Extract line numbers if present
            if " at line " in failure_reason:
                try:
                    line_num = failure_reason.split(" at line ")[1].split(":")[0]
                    analysis["line_number_patterns"][line_num] += 1
                except:
                    pass
    
    return analysis


def analyze_debug_info(debug_results: List[Dict]) -> Dict[str, Any]:
    """Analyze debug information for deeper insights."""
    debug_analysis = {
        "stages": defaultdict(int),
        "common_stderr_patterns": defaultdict(int),
        "traceback_patterns": defaultdict(int),
        "code_patterns": defaultdict(int)
    }
    
    for result in debug_results:
        debug_info = result.get('debug_info', {})
        stage = debug_info.get('stage', 'unknown')
        debug_analysis["stages"][stage] += 1
        
        # Analyze stderr patterns
        stderr = debug_info.get('stderr', '')
        if stderr:
            # Look for common patterns in stderr
            if 'Traceback' in stderr:
                debug_analysis["traceback_patterns"]['Has traceback'] += 1
            if 'ModuleNotFoundError' in stderr:
                debug_analysis["traceback_patterns"]['Module not found'] += 1
            if 'AttributeError' in stderr:
                debug_analysis["traceback_patterns"]['Attribute error'] += 1
            if 'TypeError' in stderr:
                debug_analysis["traceback_patterns"]['Type error'] += 1
            if 'AssertionError' in stderr:
                debug_analysis["traceback_patterns"]['Assertion error'] += 1
        
        # Analyze generated code patterns
        code_preview = debug_info.get('generated_code_preview', '')
        if code_preview:
            if 'import ' in code_preview:
                debug_analysis["code_patterns"]['Has imports'] += 1
            if 'def ' in code_preview:
                debug_analysis["code_patterns"]['Has function definition'] += 1
            if 'task_func' in code_preview:
                debug_analysis["code_patterns"]['Uses task_func'] += 1
            if 'func(' in code_preview:
                debug_analysis["code_patterns"]['Uses func'] += 1
    
    return debug_analysis


def print_analysis_report(analysis: Dict[str, Any], debug_analysis: Dict[str, Any] = None):
    """Print a comprehensive analysis report."""
    print("="*80)
    print("FAILURE ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total tests: {analysis['total_tests']}")
    print(f"  Passed: {analysis['passed_tests']}")
    print(f"  Failed: {analysis['failed_tests']}")
    print(f"  Pass rate: {analysis['passed_tests']/analysis['total_tests']*100:.1f}%")
    
    print(f"\nError Categories:")
    for category, errors in sorted(analysis['error_categories'].items(), 
                                 key=lambda x: len(x[1]), reverse=True):
        count = len(errors)
        percentage = count / analysis['failed_tests'] * 100 if analysis['failed_tests'] > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}% of failures)")
        
        # Show a few examples
        if count > 0:
            print(f"    Examples:")
            for i, error in enumerate(errors[:3]):
                print(f"      {i+1}. {error['task_id']}: {error['failure_reason'][:100]}...")
            if count > 3:
                print(f"      ... and {count - 3} more")
        print()
    
    if analysis['module_issues']:
        print(f"Missing Modules:")
        for module, count in sorted(analysis['module_issues'].items(), 
                                  key=lambda x: x[1], reverse=True):
            print(f"  {module}: {count} times")
    
    if analysis['specific_errors']:
        print(f"\nMost Common Specific Errors:")
        for error, count in sorted(analysis['specific_errors'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]:
            print(f"  [{count}x] {error[:120]}{'...' if len(error) > 120 else ''}")
    
    if analysis['line_number_patterns']:
        print(f"\nCommon Error Line Numbers:")
        for line_num, count in sorted(analysis['line_number_patterns'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]:
            print(f"  Line {line_num}: {count} errors")
    
    if debug_analysis:
        print(f"\nDebug Information Analysis:")
        if debug_analysis['stages']:
            print(f"  Failure stages:")
            for stage, count in debug_analysis['stages'].items():
                print(f"    {stage}: {count}")
        
        if debug_analysis['traceback_patterns']:
            print(f"  Common traceback patterns:")
            for pattern, count in sorted(debug_analysis['traceback_patterns'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"    {pattern}: {count}")
        
        if debug_analysis['code_patterns']:
            print(f"  Generated code patterns:")
            for pattern, count in sorted(debug_analysis['code_patterns'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"    {pattern}: {count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze test failure patterns")
    parser.add_argument("--results", required=True,
                       help="Path to test results JSONL file")
    parser.add_argument("--debug", 
                       help="Path to debug JSON file (optional)")
    parser.add_argument("--output",
                       help="Output file for analysis report (optional)")
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file {results_file} does not exist")
        return
    
    print(f"Loading results from: {results_file}")
    results = load_test_results(results_file)
    
    # Load debug info if available
    debug_results = []
    debug_analysis = None
    if args.debug:
        debug_file = Path(args.debug)
        if debug_file.exists():
            print(f"Loading debug info from: {debug_file}")
            debug_results = load_debug_results(debug_file)
            debug_analysis = analyze_debug_info(debug_results)
        else:
            print(f"Warning: Debug file {debug_file} does not exist")
    
    # Perform analysis
    print("Analyzing failure patterns...")
    analysis = analyze_error_patterns(results)
    
    # Print report
    print_analysis_report(analysis, debug_analysis)
    
    # Save report if requested
    if args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            # Save detailed analysis as JSON
            combined_analysis = {
                "basic_analysis": analysis,
                "debug_analysis": debug_analysis
            }
            # Convert defaultdict to regular dict for JSON serialization
            def convert_defaultdict(obj):
                if isinstance(obj, defaultdict):
                    return dict(obj)
                elif isinstance(obj, dict):
                    return {k: convert_defaultdict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_defaultdict(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_defaultdict(combined_analysis), f, indent=2)
        print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()