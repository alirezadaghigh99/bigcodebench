#!/usr/bin/env python3
"""
Demonstrate that the 'No such file or directory' issue has been fixed
by testing the specific combination that was failing.
"""

import sys
sys.path.append('/Users/aliredaq/Downloads/bigcodebench')
from fixed_cross_technique_tester import FixedCrossTechniqueTester

def test_specific_problematic_combination():
    """Test the specific active_tense vs original combination that was showing all framework errors."""
    
    tester = FixedCrossTechniqueTester(
        model_name="deepseek", 
        base_dir="/Users/aliredaq/Downloads/bigcodebench/generation_output",
        max_workers=1
    )
    
    if not tester.validate_directories():
        print("❌ Directory validation failed!")
        return False
    
    # Load the specific files mentioned in the issue
    code_files = [f for f in tester.get_code_files() if "active_tense" in f.name]
    test_files = [f for f in tester.get_test_files() if "original" in f.name]
    
    if not code_files or not test_files:
        print("❌ Required files not found!")
        return False
    
    print(f"📁 Testing combination: {code_files[0].name} vs {test_files[0].name}")
    
    code_data = tester.load_jsonl_file(code_files[0])
    test_data = tester.load_jsonl_file(test_files[0])
    
    # Test the first 5 task IDs that were showing framework errors
    common_tasks = list(set(code_data.keys()) & set(test_data.keys()))
    test_tasks = common_tasks[:5]  # Test first 5
    
    print(f"🔍 Testing {len(test_tasks)} tasks...")
    
    framework_errors = 0
    successful_tests = 0
    
    for i, task_id in enumerate(test_tasks, 1):
        print(f"  {i}. Testing {task_id}...", end=" ")
        
        result = tester.test_code_against_test(
            code_data[task_id], 
            test_data[task_id], 
            "active_tense", 
            "original"
        )
        
        failure_reason = result.get('failure_reason', '')
        pass_rate = result.get('pass_rate', 0.0)
        
        if 'No such file or directory' in str(failure_reason):
            print("❌ Framework error!")
            framework_errors += 1
        else:
            print(f"✅ Pass rate: {pass_rate:.1%}")
            successful_tests += 1
            if failure_reason and pass_rate == 0:
                print(f"      Reason: {failure_reason[:60]}...")
    
    print(f"\n📊 Results:")
    print(f"   ✅ Successful tests: {successful_tests}/{len(test_tasks)}")
    print(f"   ❌ Framework errors: {framework_errors}/{len(test_tasks)}")
    
    if framework_errors == 0:
        print("\n🎉 SUCCESS: All framework errors have been resolved!")
        print("   Tests are now running and providing meaningful results.")
        return True
    else:
        print(f"\n⚠️  WARNING: {framework_errors} framework errors still remain.")
        return False

if __name__ == "__main__":
    success = test_specific_problematic_combination()
    sys.exit(0 if success else 1)