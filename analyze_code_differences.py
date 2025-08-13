#!/usr/bin/env python3
"""
Analyze specific differences between canonical and generated code to create conversion rules.
"""

import json
import re
from pathlib import Path

def analyze_specific_differences():
    """Analyze the specific differences between canonical and generated approaches."""
    
    print("🔍 ANALYZING CODE DIFFERENCES")
    print("=" * 60)
    
    # Load canonical solution
    with open("dataset/bigcodebench.jsonl", 'r') as f:
        canonical_task = json.loads(f.readline().strip())
    
    # Load generated code
    with open("test_venv.jsonl", 'r') as f:
        generated_task = json.loads(f.readline().strip())
    
    canonical_code = canonical_task['code_prompt'] + canonical_task['canonical_solution']
    generated_code = generated_task['response_code']
    
    print(f"Task: {canonical_task['task_id']}")
    print("\n📋 KEY DIFFERENCES:")
    
    # 1. FTP connection approach
    print("\n1. FTP CONNECTION:")
    canonical_ftp = re.search(r'ftp_obj = ftplib\.FTP\([^)]+\)', canonical_code)
    generated_ftp = re.search(r'ftp = ftplib\.FTP\(\)', generated_code)
    
    if canonical_ftp and generated_ftp:
        print(f"  Canonical: {canonical_ftp.group()}")
        print(f"  Generated: {generated_ftp.group()}")
        print("  → Generated uses empty constructor, canonical passes server directly")
    
    # 2. Subprocess approach
    print("\n2. SUBPROCESS USAGE:")
    canonical_sub = re.search(r'subprocess\.call\([^)]+\)', canonical_code)
    generated_sub = re.search(r'subprocess\.run\([^)]+\)', generated_code, re.DOTALL)
    
    if canonical_sub:
        print(f"  Canonical: subprocess.call(command, shell=True)")
    if generated_sub:
        print(f"  Generated: subprocess.run(command, stdout=DEVNULL, stderr=DEVNULL, check=False)")
        print("  → Generated uses run() with output suppression, canonical uses call()")
    
    # 3. File listing approach
    print("\n3. FILE LISTING:")
    if 'ftp_obj.nlst()' in canonical_code:
        print("  Canonical: Direct ftp_obj.nlst() iteration")
    if 'ftp.size(entry)' in generated_code:
        print("  Generated: Filters files using ftp.size() check")
        print("  → Generated adds file filtering logic not in canonical")
    
    # 4. Error handling differences
    print("\n4. ERROR HANDLING:")
    canonical_tries = len(re.findall(r'try:', canonical_code))
    generated_tries = len(re.findall(r'try:', generated_code))
    print(f"  Canonical: {canonical_tries} try blocks")
    print(f"  Generated: {generated_tries} try blocks")
    
    # 5. Directory creation
    print("\n5. DIRECTORY HANDLING:")
    if 'os.makedirs' in canonical_code:
        print("  Canonical: Creates 'downloaded_files' directory")
    if 'downloaded_files' not in generated_code:
        print("  Generated: No directory creation")
        print("  → Generated missing directory creation logic")
    
    return canonical_code, generated_code

def create_conversion_rules():
    """Create conversion rules to transform generated code to canonical format."""
    
    print("\n🔧 CONVERSION RULES TO IMPLEMENT:")
    print("=" * 60)
    
    rules = [
        "1. FTP Constructor: ftplib.FTP() → ftplib.FTP(server)",
        "2. Subprocess: subprocess.run(...) → subprocess.call(command, shell=True)",
        "3. File Listing: Remove ftp.size() filtering, use direct nlst() iteration",
        "4. Directory Creation: Add os.makedirs('downloaded_files') logic",
        "5. Command Format: Use f-string format matching canonical",
        "6. Error Handling: Simplify to match canonical patterns",
        "7. Variable Names: ftp → ftp_obj",
        "8. Download Path: Add -P {download_dir} to wget command"
    ]
    
    for rule in rules:
        print(f"  {rule}")
    
    print("\n💡 APPROACH:")
    print("  • Create AST-based code transformer")
    print("  • Apply pattern matching and replacement")
    print("  • Validate syntax after transformation")
    print("  • Test against canonical solutions")

if __name__ == "__main__":
    analyze_specific_differences()
    create_conversion_rules()