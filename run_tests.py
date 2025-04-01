import pytest
import sys
import subprocess
from pathlib import Path
import argparse

def run_backend_tests(verbose: bool = False):
    """Run backend Python tests"""
    print("\n=== Running Backend Tests ===")
    args = ["-v"] if verbose else []
    pytest.main([str(Path("tests")), *args])

def run_frontend_tests(verbose: bool = False):
    """Run frontend tests"""
    print("\n=== Running Frontend Tests ===")
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("Frontend directory not found!")
        return False
    
    try:
        cmd = ["npm", "test"]
        if verbose:
            cmd.append("--verbose")
        subprocess.run(cmd, cwd=str(frontend_dir), check=True)
        return True
    except subprocess.CalledProcessError:
        print("Frontend tests failed!")
        return False

def verify_dependencies():
    """Verify that all required dependencies are installed"""
    print("\n=== Verifying Dependencies ===")
    
    # Check Python dependencies
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"],
            check=True
        )
        print("✓ Backend dependencies installed")
    except subprocess.CalledProcessError:
        print("✗ Failed to install backend dependencies")
        return False
    
    # Check Node.js dependencies
    try:
        subprocess.run(
            ["npm", "install"], 
            cwd="frontend",
            check=True
        )
        print("✓ Frontend dependencies installed")
    except subprocess.CalledProcessError:
        print("✗ Failed to install frontend dependencies")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run OCR system tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--backend-only", action="store_true", help="Run only backend tests")
    parser.add_argument("--frontend-only", action="store_true", help="Run only frontend tests")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency verification")
    
    args = parser.parse_args()
    
    if not args.skip_deps:
        if not verify_dependencies():
            sys.exit(1)
    
    success = True
    
    if not args.frontend_only:
        run_backend_tests(args.verbose)
    
    if not args.backend_only:
        if not run_frontend_tests(args.verbose):
            success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()