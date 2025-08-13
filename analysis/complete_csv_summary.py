#!/usr/bin/env python3
"""
Script to summarize the complete dual agreement CSV file with test cases.
"""

import pandas as pd
import os

def main():
    csv_file = "/Users/aliredaq/Downloads/bigcodebench/analysis/complete_dual_agreement_success_instances.csv"
    
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
    
    # Score improvement statistics
    print("\nScore improvement statistics:")
    print(f"Min improvement: {df['score_improvement'].min()}")
    print(f"Max improvement: {df['score_improvement'].max()}")
    print(f"Average improvement: {df['score_improvement'].mean():.2f}")
    print(f"Median improvement: {df['score_improvement'].median():.2f}")
    
    # Check data completeness
    print("\nData completeness check:")
    for col in df.columns:
        non_empty = df[col].notna().sum()
        print(f"  {col}: {non_empty}/{len(df)} non-empty ({non_empty/len(df)*100:.1f}%)")
    
    # Sample test case data
    print("\nSample test case data (first row):")
    if len(df) > 0:
        row = df.iloc[0]
        print(f"Task: {row['task_id']}")
        print(f"Model: {row['model']}")
        print(f"Technique: {row['technique']}")
        print(f"Score improvement: {row['score_improvement']}")
        
        print(f"\nOriginal test prompt (first 200 chars):")
        print(str(row['original_test_prompt'])[:200] + "..." if len(str(row['original_test_prompt'])) > 200 else str(row['original_test_prompt']))
        
        print(f"\nTechnique test prompt (first 200 chars):")
        print(str(row['technique_test_prompt'])[:200] + "..." if len(str(row['technique_test_prompt'])) > 200 else str(row['technique_test_prompt']))
        
        print(f"\nOriginal generated test (first 300 chars):")
        print(str(row['original_generated_test'])[:300] + "..." if len(str(row['original_generated_test'])) > 300 else str(row['original_generated_test']))
        
        print(f"\nTechnique generated test (first 300 chars):")
        print(str(row['technique_generated_test'])[:300] + "..." if len(str(row['technique_generated_test'])) > 300 else str(row['technique_generated_test']))
    
    # Top 5 improvements
    print("\nTop 5 score improvements:")
    top_improvements = df.nlargest(5, 'score_improvement')[['model', 'task_id', 'technique', 'score_improvement', 'original_score', 'chosen_score']]
    print(top_improvements.to_string(index=False))

if __name__ == "__main__":
    main()