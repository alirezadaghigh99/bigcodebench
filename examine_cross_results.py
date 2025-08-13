#!/usr/bin/env python3
"""
Examine cross-technique results with detailed code and test information.
"""

import json
from pathlib import Path

def examine_results():
    """Examine some cross-technique results with details."""
    
    # Load some results
    results_file = Path("generation_output/o4-mini/cross_result_test/detailed_results.jsonl")
    
    if not results_file.exists():
        print("Results file not found. Run test_small_cross_technique.py first.")
        return
    
    print("=== CROSS-TECHNIQUE RESULTS ANALYSIS ===\n")
    
    # Load and categorize results
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                results.append(result)
    
    # Show some examples of different failure types
    failure_types = {}
    passed_examples = []
    
    for result in results:
        failure_reason = result.get('failure_reason')
        if failure_reason:
            category = failure_reason.split(' - ')[0] if ' - ' in failure_reason else failure_reason
            if category not in failure_types:
                failure_types[category] = []
            if len(failure_types[category]) < 2:  # Keep only 2 examples per type
                failure_types[category].append(result)
        else:
            if len(passed_examples) < 2:  # Keep only 2 passed examples
                passed_examples.append(result)
    
    # Show passed examples
    if passed_examples:
        print("=== PASSED EXAMPLES ===")
        for i, result in enumerate(passed_examples, 1):
            print(f"\n--- Passed Example {i}: {result['task_id']} ---")
            print(f"Pass rate: {result['pass_rate']:.1%}")
            
            if 'executed_code' in result and result['executed_code']:
                print(f"\nExecuted Code (first 300 chars):")
                print(result['executed_code'][:300] + "..." if len(result['executed_code']) > 300 else result['executed_code'])
            
            if 'executed_test' in result and result['executed_test']:
                print(f"\nExecuted Test (first 300 chars):")
                print(result['executed_test'][:300] + "..." if len(result['executed_test']) > 300 else result['executed_test'])
    
    # Show failure examples
    print(f"\n=== FAILURE EXAMPLES ===")
    for category, examples in failure_types.items():
        print(f"\n{'='*60}")
        print(f"FAILURE TYPE: {category}")
        print(f"{'='*60}")
        
        for i, result in enumerate(examples, 1):
            print(f"\n--- Example {i}: {result['task_id']} ---")
            print(f"Detailed Error: {result.get('detailed_error', 'No details')}")
            
            if 'executed_code' in result and result['executed_code']:
                print(f"\nExecuted Code (first 200 chars):")
                print(result['executed_code'][:200] + "..." if len(result['executed_code']) > 200 else result['executed_code'])
            
            if 'executed_test' in result and result['executed_test']:
                print(f"\nExecuted Test (first 200 chars):")
                print(result['executed_test'][:200] + "..." if len(result['executed_test']) > 200 else result['executed_test'])
            
            print(f"\nFailure Reason: {result.get('failure_reason', 'Unknown')}")
    
    # Summary
    total = len(results)
    passed = len([r for r in results if r.get('pass_rate', 0) > 0])
    print(f"\n=== SUMMARY ===")
    print(f"Total results: {total}")
    print(f"Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Failed: {total - passed} ({(total-passed)/total*100:.1f}%)")
    print(f"Unique failure types: {len(failure_types)}")

if __name__ == "__main__":
    examine_results()