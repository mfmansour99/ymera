#!/usr/bin/env python3
"""
YMERA Repository Validation Script
Checks repository structure and readiness for deployment
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}!{Colors.END} {text}")

def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.END} {text}")

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    return Path(filepath).exists()

def check_directory_exists(dirpath: str) -> bool:
    """Check if a directory exists"""
    return Path(dirpath).is_dir()

def validate_structure() -> Tuple[int, int]:
    """Validate directory structure"""
    print_header("Directory Structure Validation")
    
    passed = 0
    failed = 0
    
    # Required directories
    required_dirs = [
        "app",
        "app/core",
        "app/agents",
        "app/api",
        "app/database",
        "app/services",
        "app/utils",
        "src",
        "src/components",
        "src/hooks",
        "src/services",
        "src/utils",
        "src/types",
    ]
    
    for directory in required_dirs:
        if check_directory_exists(directory):
            print_success(f"Directory exists: {directory}")
            passed += 1
        else:
            print_error(f"Missing directory: {directory}")
            failed += 1
    
    return passed, failed

def validate_config_files() -> Tuple[int, int]:
    """Validate configuration files"""
    print_header("Configuration Files Validation")
    
    passed = 0
    failed = 0
    
    required_files = [
        ".gitignore",
        ".env.example",
        "requirements.txt",
        "package.json",
        "tsconfig.json",
        "Dockerfile",
        "docker-compose.yml",
        "README.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "DEPLOYMENT.md",
    ]
    
    for filepath in required_files:
        if check_file_exists(filepath):
            print_success(f"File exists: {filepath}")
            passed += 1
        else:
            print_error(f"Missing file: {filepath}")
            failed += 1
    
    return passed, failed

def validate_main_files() -> Tuple[int, int]:
    """Validate main application files"""
    print_header("Main Application Files Validation")
    
    passed = 0
    failed = 0
    
    required_files = [
        "main_production.py",
        "main.py",
        "app/__init__.py",
    ]
    
    for filepath in required_files:
        if check_file_exists(filepath):
            print_success(f"File exists: {filepath}")
            passed += 1
        else:
            print_warning(f"Optional file missing: {filepath}")
            failed += 1
    return passed, failed

def check_python_syntax() -> Tuple[int, int]:
    """Check Python files for syntax errors"""
    print_header("Python Syntax Validation")
    
    passed = 0
    failed = 0
    
    python_files = [
        "main_production.py",
        "main.py",
        "test_platform.py",
    ]
    
    for filepath in python_files:
        if check_file_exists(filepath):
            try:
                import py_compile
                py_compile.compile(filepath, doraise=True)
                print_success(f"Valid Python syntax: {filepath}")
                passed += 1
            except Exception as e:
                print_error(f"Syntax error in {filepath}: {str(e)}")
                failed += 1
    
    return passed, failed

def check_package_files() -> Tuple[int, int]:
    """Check if __init__.py files exist in packages"""
    print_header("Python Package Validation")
    
    passed = 0
    failed = 0
    
    package_dirs = [
        "app",
        "app/core",
        "app/agents",
        "app/api",
        "app/database",
        "app/services",
        "app/utils",
        "app/monitoring",
    ]
    
    for directory in package_dirs:
        init_file = Path(directory) / "__init__.py"
        if init_file.exists():
            print_success(f"Package file exists: {init_file}")
            passed += 1
        else:
            print_error(f"Missing __init__.py in: {directory}")
            failed += 1
    
    return passed, failed

def check_security() -> Tuple[int, int]:
    """Check for security issues"""
    print_header("Security Validation")
    
    passed = 0
    warnings = 0
    
    # Check for .env file (should not be committed)
    if check_file_exists(".env"):
        print_warning(".env file exists (should not be committed to git)")
        warnings += 1
    else:
        print_success(".env file not found (good - use .env.example)")
        passed += 1
    
    # Check for sensitive file patterns
    sensitive_patterns = [
        "*.key",
        "*.pem",
        "*.p12",
        "*.pfx",
    ]
    
    import glob
    found_sensitive = False
    for pattern in sensitive_patterns:
        matches = glob.glob(pattern)
        if matches:
            for match in matches:
                print_warning(f"Sensitive file found: {match}")
                found_sensitive = True
                warnings += 1
    
    if not found_sensitive:
        print_success("No sensitive files detected")
        passed += 1
    
    return passed, warnings

def generate_report(results: dict):
    """Generate final validation report"""
    print_header("Validation Report")
    
    total_passed = sum(r[0] for r in results.values())
    total_failed = sum(r[1] for r in results.values())
    total_checks = total_passed + total_failed
    
    print(f"Total Checks: {total_checks}")
    print_success(f"Passed: {total_passed}")
    
    if total_failed > 0:
        print_error(f"Failed: {total_failed}")
    else:
        print(f"Failed: {total_failed}")
    
    if total_failed == 0:
        print(f"\n{Colors.GREEN}{'=' * 70}{Colors.END}")
        print(f"{Colors.GREEN}✓ All validations passed! Repository is ready for deployment.{Colors.END}")
        print(f"{Colors.GREEN}{'=' * 70}{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{'=' * 70}{Colors.END}")
        print(f"{Colors.YELLOW}! Some validations failed. Please review the issues above.{Colors.END}")
        print(f"{Colors.YELLOW}{'=' * 70}{Colors.END}")
        return 1

def main():
    """Main validation function"""
    print_header("YMERA Repository Validation")
    print("Checking repository structure and deployment readiness...\n")
    
    results = {
        'structure': validate_structure(),
        'config': validate_config_files(),
        'main': validate_main_files(),
        'syntax': check_python_syntax(),
        'packages': check_package_files(),
        'security': check_security(),
    }
    
    exit_code = generate_report(results)
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Review any failed checks above")
    print("  2. Run './start.sh' for quick start")
    print("  3. See DEPLOYMENT.md for production deployment")
    print("=" * 70 + "\n")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
