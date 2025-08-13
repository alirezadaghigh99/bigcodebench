#!/usr/bin/env python3
"""
Investigate 10 specific instances to verify cross-technique testing framework works properly.
"""

import json
from pathlib import Path
from fixed_cross_technique_tester import FixedCrossTechniqueTester
import tempfile

def investigate_framework():
    """Test framework on 10 specific instances to verify it works properly."""
    
    print("=== INVESTIGATING CROSS-TECHNIQUE FRAMEWORK ===\n")
    
    # Use o4-mini model
    model_dir = Path("generation_output/o4-mini")
    output_dir = model_dir / "cross_result_investigation"
    
    # Initialize tester
    tester = FixedCrossTechniqueTester(
        code_dir=str(model_dir),
        test_dir=str(model_dir), 
        output_dir=str(output_dir),
        max_workers=1
    )
    
    # Get files
    code_files = tester.get_code_files()
    test_files = tester.get_test_files()
    
    if not code_files or not test_files:
        print("No files found")
        return
    
    # Test first combination (same as previous test but with detailed analysis)
    code_file = code_files[0]
    test_file = test_files[0]
    
    print(f"Testing: {code_file.name} vs {test_file.name}")
    
    # Load data
    code_data = tester.load_jsonl_file(code_file)
    test_data = tester.load_jsonl_file(test_file)
    
    # Find common task IDs
    common_tasks = set(code_data.keys()) & set(test_data.keys())
    
    if len(common_tasks) < 10:
        print(f"Warning: Only {len(common_tasks)} common tasks found, will test all available")
    
    # Take first 10 common tasks
    sample_tasks = sorted(list(common_tasks))[:10]
    
    print(f"\nInvestigating {len(sample_tasks)} tasks:\n")
    
    detailed_results = []
    
    for i, task_id in enumerate(sample_tasks, 1):
        print(f"--- INSTANCE {i}: {task_id} ---")
        
        code_entry = code_data[task_id]
        test_entry = test_data[task_id]
        
        # Get raw code and test
        raw_code = code_entry.get('response_code', '')
        raw_test = test_entry.get('response_code', '')
        
        print(f"Raw code (first 100 chars): {raw_code[:100]}...")
        print(f"Raw test (first 100 chars): {raw_test[:100]}...")
        
        # Test the conversion process step by step
        try:
            # Step 1: Convert code function to task_func
            from fixed_cross_technique_tester import convert_code_function_to_task_func
            converted_code = convert_code_function_to_task_func(raw_code)
            print(f"✅ Code conversion successful")
            
            # Step 2: Clean test code (including removing problematic imports)
            cleaned_test = tester.clean_test_code(raw_test)
            print(f"✅ Test cleaning successful")
            
            # Check if problematic imports were removed
            if 'from task import task_func' in cleaned_test:
                print(f"⚠️  WARNING: 'from task import task_func' still present in cleaned test")
            else:
                print(f"✅ Problematic imports removed")
            
            # Step 3: Convert test function calls to task_func
            from fixed_cross_technique_tester import convert_test_function_calls_to_task_func
            converted_test = convert_test_function_calls_to_task_func(cleaned_test)
            print(f"✅ Test conversion successful")
            
            # Step 4: Check compatibility
            compatible = tester.check_compatibility(converted_code, converted_test)
            print(f"Compatibility check: {'✅ PASS' if compatible else '❌ FAIL'}")
            
            if compatible:
                # Step 5: Run actual test
                result = tester.test_code_against_test(
                    code_entry, test_entry, 
                    "investigation_code", "investigation_test"
                )
                
                pass_rate = result.get('pass_rate', 0.0)
                failure_reason = result.get('failure_reason')
                
                print(f"Test execution: {'✅ PASS' if pass_rate > 0 else '❌ FAIL'}")
                if pass_rate > 0:
                    print(f"Pass rate: {pass_rate:.1%}")
                else:
                    print(f"Failure reason: {failure_reason}")
                
                detailed_results.append({
                    'task_id': task_id,
                    'conversion_success': True,
                    'compatibility_check': compatible,
                    'test_execution_success': pass_rate > 0,
                    'pass_rate': pass_rate,
                    'failure_reason': failure_reason
                })
            else:
                print(f"❌ Skipped test execution due to incompatibility")
                detailed_results.append({
                    'task_id': task_id,
                    'conversion_success': True,
                    'compatibility_check': False,
                    'test_execution_success': False,
                    'pass_rate': 0.0,
                    'failure_reason': 'Incompatible code/test pair'
                })
        
        except Exception as e:
            print(f"❌ Error in processing: {str(e)}")
            detailed_results.append({
                'task_id': task_id,
                'conversion_success': False,
                'compatibility_check': False,
                'test_execution_success': False,
                'pass_rate': 0.0,
                'failure_reason': f'Processing error: {str(e)}'
            })
        
        print()  # Empty line for readability
    
    # Summary
    print("=== INVESTIGATION SUMMARY ===")
    conversion_success_count = sum(1 for r in detailed_results if r['conversion_success'])
    compatibility_pass_count = sum(1 for r in detailed_results if r['compatibility_check'])
    execution_success_count = sum(1 for r in detailed_results if r['test_execution_success'])
    
    print(f"Conversion success: {conversion_success_count}/{len(detailed_results)} ({conversion_success_count/len(detailed_results)*100:.1f}%)")
    print(f"Compatibility check pass: {compatibility_pass_count}/{len(detailed_results)} ({compatibility_pass_count/len(detailed_results)*100:.1f}%)")
    print(f"Test execution success: {execution_success_count}/{len(detailed_results)} ({execution_success_count/len(detailed_results)*100:.1f}%)")
    
    # Show failure reasons
    failure_types = {}
    for result in detailed_results:
        if not result['test_execution_success']:
            reason = result['failure_reason']
            failure_types[reason] = failure_types.get(reason, 0) + 1
    
    if failure_types:
        print(f"\nFailure breakdown:")
        for reason, count in sorted(failure_types.items()):
            print(f"  {reason}: {count}")
    
    # Save detailed results
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "investigation_results.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Overall assessment
    print(f"\n=== FRAMEWORK ASSESSMENT ===")
    if conversion_success_count == len(detailed_results):
        print("✅ Code/test conversion pipeline working correctly")
    else:
        print("❌ Issues found in code/test conversion pipeline")
    
    if compatibility_pass_count > 0:
        print("✅ Compatibility checking working")
    else:
        print("❌ All samples failed compatibility check")
    
    if execution_success_count > 0:
        print("✅ Test execution pipeline working")
        print(f"Framework appears to be working with {execution_success_count/len(detailed_results)*100:.1f}% success rate on investigated samples")
    else:
        print("❌ No tests executed successfully")
        print("Framework may need further debugging")

if __name__ == "__main__":
    investigate_framework()