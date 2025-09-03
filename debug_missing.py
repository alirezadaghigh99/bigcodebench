#!/usr/bin/env python3
"""Debug script to see exactly which files are being marked as missing."""

import sys
sys.path.append('/Users/aliredaq/Downloads/bigcodebench')
from test_generated_code_pipeline import TestGeneratedCodePipeline

# Create pipeline instance
pipeline = TestGeneratedCodePipeline('deepseek', '/Users/aliredaq/Downloads/bigcodebench/generation_output', resume=True)

# Validate directories to set up actual_model_name
pipeline.validate_directories()

# Get the missing files
missing_files = pipeline.get_missing_test_files()

print(f"Found {len(missing_files)} missing files:")
for i, f in enumerate(missing_files, 1):
    print(f"{i:2d}. {f}")