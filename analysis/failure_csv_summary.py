#!/usr/bin/env python3
"""
Script to summarize the failure case analysis CSV file.
"""

import pandas as pd
import os

def main():
    csv_file = "/Users/aliredaq/Downloads/bigcodebench/analysis/failure_case_analysis.csv"
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Count by model
    print("\nCount by model:")
    model_counts = df['model'].value_counts()
    print(model_counts)
    
    # Count by technique
    print("\nCount by technique:")
    technique_counts = df['technique'].value_counts()
    print(technique_counts)
    
    # Count by chosen solution
    print("\nCount by chosen_solution:")
    chosen_counts = df['chosen_solution'].value_counts()
    print(chosen_counts)
    
    # Score difference statistics
    print("\nScore difference statistics (chosen - original):")
    print(f"Min difference: {df['score_difference'].min()}")
    print(f"Max difference: {df['score_difference'].max()}")
    print(f"Average difference: {df['score_difference'].mean():.2f}")
    print(f"Median difference: {df['score_difference'].median():.2f}")
    
    # Score difference distribution
    negative_count = (df['score_difference'] < 0).sum()
    zero_count = (df['score_difference'] == 0).sum()
    positive_count = (df['score_difference'] > 0).sum()
    
    print(f"\nScore difference distribution:")
    print(f"  Negative (chosen < original): {negative_count}")
    print(f"  Zero (chosen = original): {zero_count}")
    print(f"  Positive (chosen > original): {positive_count}")
    
    # Check actual_result and original_test_result
    print(f"\nVerification:")
    print(f"All entries have actual_result = 0: {(df['actual_result'] == 0).all()}")
    print(f"All entries have original_test_result = 1: {(df['original_test_result'] == 1).all()}")
    
    # Technique test result distribution
    print(f"\nTechnique test result distribution:")
    print(df['technique_test_result'].value_counts())
    
    # Cases where technique test also failed
    technique_failed = df['technique_test_result'] == 0
    print(f"\nCases where technique test also failed: {technique_failed.sum()}")
    
    # Cases where score actually improved despite system failure
    improved_cases = df[df['score_difference'] > 0]
    print(f"\nCases where score improved despite system failure: {len(improved_cases)}")
    if len(improved_cases) > 0:
        print("Details:")
        for _, row in improved_cases.iterrows():
            print(f"  {row['model']}, {row['task_id']}, {row['technique']}: +{row['score_difference']} points")
    
    # Sample case analysis
    print(f"\nSample failure case (first row):")
    if len(df) > 0:
        row = df.iloc[0]
        print(f"Task: {row['task_id']}")
        print(f"Model: {row['model']}")
        print(f"Technique: {row['technique']}")
        print(f"Chosen solution: {row['chosen_solution']}")
        print(f"Score difference: {row['score_difference']}")
        print(f"Original test result: {row['original_test_result']}")
        print(f"Technique test result: {row['technique_test_result']}")
        print(f"System actual result: {row['actual_result']}")

if __name__ == "__main__":
    main()