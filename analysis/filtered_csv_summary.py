#!/usr/bin/env python3
"""
Script to summarize the filtered dual agreement CSV file.
"""

import pandas as pd
import os

def main():
    csv_file = "/Users/aliredaq/Downloads/bigcodebench/analysis/filtered_dual_agreement_success_instances.csv"
    
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
    
    # Score improvement statistics
    print("\nScore improvement statistics:")
    print(f"Min improvement: {df['score_improvement'].min()}")
    print(f"Max improvement: {df['score_improvement'].max()}")
    print(f"Average improvement: {df['score_improvement'].mean():.2f}")
    print(f"Median improvement: {df['score_improvement'].median():.2f}")
    
    # Top 10 improvements
    print("\nTop 10 score improvements:")
    top_improvements = df.nlargest(10, 'score_improvement')[['model', 'task_id', 'technique', 'score_improvement']]
    print(top_improvements.to_string(index=False))
    
    # Check that chosen_score > original_score for all entries
    print(f"\nVerification - All entries have chosen_score > original_score: {(df['chosen_score'] > df['original_score']).all()}")
    
    # Check that all entries have actual_result = 1
    print(f"All entries have actual_result = 1: {(df['actual_result'] == 1).all()}")
    
    # Check original_test_result distribution
    print(f"\nOriginal test result distribution:")
    print(df['original_test_result'].value_counts())

if __name__ == "__main__":
    main()