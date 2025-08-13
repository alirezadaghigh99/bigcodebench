#!/usr/bin/env python3

import json
import re
import ast
from typing import Optional

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from response, handling various formats."""
    if not response:
        return None
    
    # Try to extract code from markdown blocks first
    code_block_pattern = r'```python\n(.*?)```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Take the last match
    
    # If no markdown blocks, try to extract from plain text
    # Look for import statements as start indicators
    lines = response.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Start collecting when we see import or def
        if re.match(r'^\s*(import|from|def)', line):
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip() if code_lines else None

def fix_truncated_code(code: str) -> str:
    """Try to fix common truncation issues."""
    if not code:
        return code
    
    # Fix unterminated triple quotes
    triple_quote_count = code.count("'''")
    if triple_quote_count % 2 == 1:  # Odd number means unterminated
        code += "\n    '''"
    
    double_triple_quote_count = code.count('"""')
    if double_triple_quote_count % 2 == 1:
        code += '\n    """'
    
    # Check if we have incomplete function body
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('def ') and line.endswith(':'):
            # Check if there's any actual function body after this
            has_body = False
            for j in range(i+1, len(lines)):
                if lines[j].strip() and not lines[j].startswith('    '):
                    break
                if lines[j].strip() and lines[j].startswith('    '):
                    has_body = True
                    break
            
            if not has_body:
                # Add a minimal body
                code += "\n    pass"
                break
    
    return code

def validate_and_fix_code(response_code: str, response: str) -> str:
    """Validate and fix code, falling back to response if needed."""
    
    # First try the response_code as is
    try:
        compile(response_code, '<string>', 'exec')
        return response_code
    except SyntaxError:
        pass
    
    # Try to fix truncation issues
    fixed_code = fix_truncated_code(response_code)
    try:
        compile(fixed_code, '<string>', 'exec')
        return fixed_code
    except SyntaxError:
        pass
    
    # Try to extract from the full response
    extracted_code = extract_code_from_response(response)
    if extracted_code:
        try:
            compile(extracted_code, '<string>', 'exec')
            return extracted_code
        except SyntaxError:
            # Try fixing the extracted code too
            fixed_extracted = fix_truncated_code(extracted_code)
            try:
                compile(fixed_extracted, '<string>', 'exec')
                return fixed_extracted
            except SyntaxError:
                pass
    
    # If all else fails, return the original (will likely fail tests but at least we track it)
    return response_code

def test_gemini_fix():
    """Test the fix on problematic Gemini entries."""
    file_path = "/Users/aliredaq/Downloads/bigcodebench/generation_output/gemini/code_active_to_passive_gemini_bigcodebench_output_llm.jsonl"
    
    print("Testing fixes on first 5 entries...")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
                
            sample = json.loads(line)
            task_id = sample['task_id']
            response_code = sample.get('response_code', '')
            response = sample.get('response', '')
            
            print(f"\n=== Entry {i+1}: {task_id} ===")
            print(f"Original code valid: ", end="")
            try:
                compile(response_code, '<string>', 'exec')
                print("✓")
                continue
            except SyntaxError:
                print("✗")
            
            # Try to fix it
            fixed_code = validate_and_fix_code(response_code, response)
            print(f"Fixed code valid: ", end="")
            try:
                compile(fixed_code, '<string>', 'exec')
                print("✓")
                print(f"Fixed code length: {len(fixed_code)}")
            except SyntaxError as e:
                print(f"✗ Still invalid: {e}")

if __name__ == "__main__":
    test_gemini_fix()