#!/usr/bin/env python3
"""
Check the original test file to see if it has proper task specifications.
"""

import json
from pathlib import Path

def check_original_test():
    """Check if the original test file has proper task context."""
    
    test_file = Path("generation_output/o4-mini/test_original_o4-mini_bigcodebench_output_llm1.jsonl")
    
    if not test_file.exists():
        print("Original test file not found")
        return
    
    print("Checking original test file...")
    
    # Load a few entries
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Only check first 3
                break
            if line.strip():
                try:
                    entry = json.loads(line)
                    task_id = entry['task_id']
                    
                    print(f"\n{'='*50}")
                    print(f"TASK ID: {task_id}")
                    print(f"{'='*50}")
                    
                    # Check prompt
                    prompt = entry.get('prompt', [])
                    print(f"Prompt messages: {len(prompt)}")
                    
                    for j, msg in enumerate(prompt):
                        if isinstance(msg, dict) and 'content' in msg:
                            content = msg['content']
                            print(f"\nMessage {j}:")
                            print(f"  {content[:300]}...")
                    
                    # Check response
                    response_code = entry.get('response_code', '')
                    print(f"\nGenerated test (first 200 chars):")
                    print(f"  {response_code[:200]}...")
                    
                except Exception as e:
                    print(f"Error parsing line {i+1}: {e}")

if __name__ == "__main__":
    check_original_test()