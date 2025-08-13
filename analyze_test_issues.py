#!/usr/bin/env python3
"""
Diagnostic script to analyze test issues and compare with BigCodeBench expectations.
Identifies the root causes of poor performance.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any


def load_dataset(dataset_path: Path) -> Dict[str, Dict]:
    """Load the original BigCodeBench dataset."""
    dataset = {}
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                task = json.loads(line.strip())
                dataset[task['task_id']] = task
    return dataset


def load_results(results_path: Path) -> List[Dict]:
    """Load test results."""
    results = []
    with open(results_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results


def load_generated_code(code_path: Path) -> Dict[str, Dict]:
    """Load generated code for comparison."""
    generated = {}
    with open(code_path, 'r') as f:
        for line in f:
            if line.strip():
                code_data = json.loads(line.strip())
                generated[code_data['task_id']] = code_data
    return generated


def analyze_failure_patterns(results: List[Dict], dataset: Dict[str, Dict], generated: Dict[str, Dict]) -> Dict[str, Any]:
    """Analyze failure patterns in detail."""
    analysis = {
        "total_tests": len(results),
        "passed": 0,
        "failed": 0,
        "framework_errors": 0,
        "failure_categories": defaultdict(list),
        "function_name_issues": [],
        "import_issues": [],
        "logic_errors": [],
        "test_assertion_failures": [],
        "detailed_breakdown": defaultdict(int)
    }
    
    for result in results:
        task_id = result['id']
        
        if result['test_result'] == 1:
            analysis["passed"] += 1
        else:
            analysis["failed"] += 1
            failure_reason = result.get('failure_reason', 'Unknown')
            
            # Categorize failures
            if 'Framework error' in failure_reason:
                analysis["framework_errors"] += 1
                analysis["failure_categories"]["framework"].append({
                    "task_id": task_id,
                    "reason": failure_reason
                })
            elif 'Syntax error' in failure_reason:
                analysis["failure_categories"]["syntax"].append({
                    "task_id": task_id,
                    "reason": failure_reason
                })
            elif 'Missing module' in failure_reason:
                module_name = failure_reason.split("'")[1] if "'" in failure_reason else "unknown"
                analysis["import_issues"].append({
                    "task_id": task_id,
                    "module": module_name,
                    "reason": failure_reason
                })
            elif 'Test failure' in failure_reason:
                analysis["test_assertion_failures"].append({
                    "task_id": task_id,
                    "reason": failure_reason
                })
                
                # Check if it's a logic error by examining the test failure
                if task_id in dataset and task_id in generated:
                    original_task = dataset[task_id]
                    gen_code = generated[task_id]
                    
                    # Look for common logic error patterns
                    if 'AssertionError' in failure_reason:
                        if 'Lists differ' in failure_reason or 'differ' in failure_reason:
                            analysis["logic_errors"].append({
                                "task_id": task_id,
                                "type": "output_mismatch",
                                "reason": failure_reason,
                                "function_name": extract_function_name_from_code(gen_code.get('response_code', ''))
                            })
            
            # Detailed breakdown
            if 'Syntax error' in failure_reason:
                analysis["detailed_breakdown"]["syntax_errors"] += 1
            elif 'Attribute error' in failure_reason:
                analysis["detailed_breakdown"]["attribute_errors"] += 1
            elif 'Type error' in failure_reason:
                analysis["detailed_breakdown"]["type_errors"] += 1
            elif 'Name error' in failure_reason:
                analysis["detailed_breakdown"]["name_errors"] += 1
            elif 'Missing module' in failure_reason:
                analysis["detailed_breakdown"]["missing_modules"] += 1
            elif 'Test failure' in failure_reason:
                analysis["detailed_breakdown"]["test_failures"] += 1
            elif 'Framework error' in failure_reason:
                analysis["detailed_breakdown"]["framework_errors"] += 1
            else:
                analysis["detailed_breakdown"]["other_errors"] += 1
    
    return analysis


def extract_function_name_from_code(code: str) -> str:
    """Extract function name from generated code."""
    if not code:
        return "unknown"
    
    import re
    pattern = r'^def\s+(\w+)\s*\('
    match = re.search(pattern, code, re.MULTILINE)
    return match.group(1) if match else "unknown"


def compare_with_leaderboard_expectations(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Compare results with expected BigCodeBench performance."""
    total = analysis["total_tests"]
    passed = analysis["passed"]
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    # Expected performance ranges for different models
    expected_ranges = {
        "gpt-4o": {"min": 50, "max": 60},
        "gpt-4o-mini": {"min": 35, "max": 45},
        "claude-3.5-sonnet": {"min": 40, "max": 50},
        "gemini-pro": {"min": 30, "max": 40}
    }
    
    comparison = {
        "current_pass_rate": pass_rate,
        "expected_range_gpt4o": expected_ranges["gpt-4o"],
        "expected_range_gpt4o_mini": expected_ranges["gpt-4o-mini"],
        "performance_gap": {},
        "likely_issues": []
    }
    
    # Calculate gaps
    for model, range_dict in expected_ranges.items():
        gap = range_dict["min"] - pass_rate
        comparison["performance_gap"][model] = gap
    
    # Identify likely issues
    if analysis["framework_errors"] > total * 0.05:  # >5% framework errors
        comparison["likely_issues"].append("High framework error rate - testing infrastructure issues")
    
    if analysis["detailed_breakdown"]["test_failures"] > total * 0.3:  # >30% test failures
        comparison["likely_issues"].append("High test failure rate - likely logic errors in generated code")
    
    if analysis["detailed_breakdown"]["missing_modules"] > total * 0.1:  # >10% missing modules
        comparison["likely_issues"].append("Missing module issues - environment setup problems")
    
    if analysis["detailed_breakdown"]["syntax_errors"] > total * 0.05:  # >5% syntax errors
        comparison["likely_issues"].append("Syntax errors - standardization process issues")
    
    return comparison


def generate_recommendations(analysis: Dict[str, Any], comparison: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations to improve performance."""
    recommendations = []
    total = analysis["total_tests"]
    
    # Framework errors
    if analysis["framework_errors"] > 0:
        recommendations.append(
            f"🔧 Fix framework errors ({analysis['framework_errors']} tests): "
            "Update error handling in test runner to prevent directory cleanup issues"
        )
    
    # Test failures (logic errors)
    test_failures = analysis["detailed_breakdown"]["test_failures"]
    if test_failures > total * 0.2:
        recommendations.append(
            f"🧠 Address logic errors ({test_failures} tests): "
            "Generated code produces wrong outputs. Consider:\n"
            "   - Better prompt engineering\n"
            "   - Model fine-tuning\n"
            "   - Post-processing validation"
        )
    
    # Missing modules
    missing_modules = analysis["detailed_breakdown"]["missing_modules"]
    if missing_modules > 0:
        module_list = list(set([item["module"] for item in analysis["import_issues"]]))
        recommendations.append(
            f"📦 Install missing modules ({missing_modules} tests): "
            f"Add to environment: {', '.join(module_list[:10])}"
        )
    
    # Syntax errors
    syntax_errors = analysis["detailed_breakdown"]["syntax_errors"]
    if syntax_errors > 0:
        recommendations.append(
            f"⚠️  Fix syntax errors ({syntax_errors} tests): "
            "Improve standardization process to avoid breaking code syntax"
        )
    
    # Performance gap
    gap = comparison["performance_gap"].get("gpt-4o-mini", 0)
    if gap > 10:
        recommendations.append(
            f"📈 Performance gap: {gap:.1f}% below expected range\n"
            "   - Consider using better base model\n"
            "   - Improve code generation prompts\n"
            "   - Add code validation before testing"
        )
    
    return recommendations


def print_detailed_report(analysis: Dict[str, Any], comparison: Dict[str, Any], recommendations: List[str]):
    """Print a comprehensive diagnostic report."""
    print("🔍 BigCodeBench Test Issues Analysis")
    print("=" * 80)
    
    # Overall stats
    total = analysis["total_tests"]
    passed = analysis["passed"]
    failed = analysis["failed"]
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Pass rate: {pass_rate:.1f}%")
    
    # Performance comparison
    print(f"\n📈 Performance Comparison:")
    print(f"  Current: {pass_rate:.1f}%")
    print(f"  Expected GPT-4o: {comparison['expected_range_gpt4o']['min']}-{comparison['expected_range_gpt4o']['max']}%")
    print(f"  Expected GPT-4o-mini: {comparison['expected_range_gpt4o_mini']['min']}-{comparison['expected_range_gpt4o_mini']['max']}%")
    print(f"  Gap from GPT-4o-mini: {comparison['performance_gap']['gpt-4o-mini']:.1f}%")
    
    # Detailed breakdown
    print(f"\n🔍 Failure Breakdown:")
    for category, count in sorted(analysis["detailed_breakdown"].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / failed * 100) if failed > 0 else 0
            print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}% of failures)")
    
    # Top issues
    if analysis["logic_errors"]:
        print(f"\n🧠 Logic Error Examples:")
        for i, error in enumerate(analysis["logic_errors"][:3], 1):
            print(f"  {i}. {error['task_id']}: {error['type']}")
            print(f"     Function: {error['function_name']}")
    
    if analysis["import_issues"]:
        print(f"\n📦 Missing Modules:")
        module_counts = Counter([item["module"] for item in analysis["import_issues"]])
        for module, count in module_counts.most_common(5):
            print(f"  {module}: {count} tests")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Framework errors details
    if analysis["framework_errors"] > 0:
        print(f"\n🔧 Framework Error Details:")
        framework_errors = analysis["failure_categories"]["framework"]
        error_types = Counter([err["reason"] for err in framework_errors])
        for error_type, count in error_types.most_common(3):
            print(f"  {error_type}: {count} occurrences")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze BigCodeBench test issues")
    parser.add_argument("--results", required=True,
                       help="Path to test results JSONL file")
    parser.add_argument("--dataset", default="dataset/bigcodebench.jsonl",
                       help="Path to BigCodeBench dataset")
    parser.add_argument("--generated", 
                       help="Path to generated code JSONL file")
    parser.add_argument("--output",
                       help="Output file for analysis report (JSON)")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    results = load_results(Path(args.results))
    dataset = load_dataset(Path(args.dataset))
    
    generated = {}
    if args.generated:
        generated = load_generated_code(Path(args.generated))
    
    # Analyze
    print("Analyzing failure patterns...")
    analysis = analyze_failure_patterns(results, dataset, generated)
    comparison = compare_with_leaderboard_expectations(analysis)
    recommendations = generate_recommendations(analysis, comparison)
    
    # Report
    print_detailed_report(analysis, comparison, recommendations)
    
    # Save detailed analysis
    if args.output:
        output_data = {
            "analysis": analysis,
            "comparison": comparison,
            "recommendations": recommendations
        }
        
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(v) for v in obj]
            else:
                return obj
        
        with open(args.output, 'w') as f:
            json.dump(convert_sets(output_data), f, indent=2)
        print(f"\n📄 Detailed analysis saved to: {args.output}")


if __name__ == "__main__":
    main()