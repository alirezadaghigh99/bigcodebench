#!/usr/bin/env python3
"""Debug script to understand file name matching patterns."""

from pathlib import Path

# Get standardized files
standardized_dir = Path("/Users/aliredaq/Downloads/bigcodebench/generation_output/deepseek_standardized")
standardized_files = sorted([f.name for f in standardized_dir.glob("code_*.jsonl")])

# Get result files  
results_dir = Path("/Users/aliredaq/Downloads/bigcodebench/test_result5/deepseek_results")
result_files = sorted([f.name for f in results_dir.glob("code_*.jsonl")])

print("=== STANDARDIZED FILES ===")
for i, f in enumerate(standardized_files):
    print(f"{i+1:2d}. {f}")

print("\n=== RESULT FILES ===")  
for i, f in enumerate(result_files):
    print(f"{i+1:2d}. {f}")

print("\n=== MAPPING ANALYSIS ===")
print("Let's see which standardized files have corresponding results:")

for std_file in standardized_files:
    print(f"\nStandardized: {std_file}")
    
    # Try to find matching result
    std_parts = std_file.replace('.jsonl', '').split('_')
    if len(std_parts) >= 4:
        technique = std_parts[1]  # First word of technique
        
        matches = []
        for result_file in result_files:
            if technique in result_file and "deepseek" in result_file:
                matches.append(result_file)
        
        if matches:
            print(f"  -> Possible matches: {matches}")
        else:
            print(f"  -> NO MATCH FOUND for technique '{technique}'")