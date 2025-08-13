#!/usr/bin/env python3
"""
Script to summarize the contents of the dual agreement CSV file.
"""

import pandas as pd
import os

def main():
    csv_file = "/Users/aliredaq/Downloads/bigcodebench/analysis/dual_agreement_success_instances.csv"
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    # Count by model
    print("\nCount by model:")
    model_counts = df['model'].value_counts()
    print(model_counts)
    
    # Count by technique
    print("\nCount by technique:")
    technique_counts = df['technique'].value_counts()
    print(technique_counts)
    
    # Count by chosen_solution
    print("\nCount by chosen_solution:")
    chosen_counts = df['chosen_solution'].value_counts()
    print(chosen_counts)
    
    # Check actual_result values
    print("\nActual result values:")
    actual_counts = df['actual_result'].value_counts()
    print(actual_counts)
    
    # Sample of the data
    print("\nFirst 3 rows summary:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"Row {i+1}: {row['model']}, {row['task_id']}, {row['technique']}")
        print(f"  Chosen: {row['chosen_solution']}, Result: {row['actual_result']}")
        print(f"  Original test: {row['original_test_result']}, Technique test: {row['technique_test_result']}")
        print()

if __name__ == "__main__":
    main()