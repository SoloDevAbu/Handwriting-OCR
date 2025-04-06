import subprocess
import sys
import time
import os
from pathlib import Path
import argparse
import signal
import platform

def check_python_dependencies():
    """Check if required Python dependencies are installed, and install if missing"""
    print("Checking Python dependencies...")
    
    required_packages = [
        "fastapi", "uvicorn", "python-multipart", "pillow", "numpy", 
        "opencv-python", "tensorflow", "sklearn", "matplotlib",
        "python-dotenv", "pydantic", "loguru"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_").split("==")[0])
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing Python packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Python dependencies installed successfully.")
    else:
        print("All Python dependencies are already installed.")

def check_node_dependencies():
    """Check if Node.js and npm are installed, and set up frontend dependencies"""
    print("Checking Node.js dependencies...")
    
    # Check if npm is available
    try:
        subprocess.check_call(["npm", "--version"], stdout=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: npm is not installed or not in PATH. Please install Node.js and npm.")
        return False
    
    # Check if node_modules exists in client directory
    client_dir = Path("client")
    if not (client_dir / "node_modules").exists():
        print("Installing frontend dependencies (this may take a while)...")
        try:
            subprocess.check_call(["npm", "install"], cwd=str(client_dir))
            print("Frontend dependencies installed successfully.")
        except subprocess.CalledProcessError:
            print("ERROR: Failed to install frontend dependencies.")
            return False
    
    return True

def create_directory_structure():
    """Create required directories if they don't exist"""
    # Create training directory
    Path("backend/training").mkdir(exist_ok=True, parents=True)
    # Create logs directory
    Path("backend/logs").mkdir(exist_ok=True, parents=True)

def start_backend():
    """Start the FastAPI backend server"""
    print("Starting backend server...")
    backend_path = Path("backend")
    env = os.environ.copy()
    
    python_executable = sys.executable
    
    return subprocess.Popen(
        [python_executable, "src/main.py"],
        cwd=str(backend_path),
        env=env
    )

def start_frontend():
    """Start the Next.js frontend development server"""
    print("Starting frontend development server...")
    frontend_path = Path("client")
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_path)
    )

def main():
    parser = argparse.ArgumentParser(description="Start OCR system services")
    parser.add_argument("--backend-only", action="store_true", help="Start only the backend server")
    parser.add_argument("--frontend-only", action="store_true", help="Start only the frontend server")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency checking")
    args = parser.parse_args()

    # Create necessary directories
    create_directory_structure()

    # Check dependencies unless explicitly skipped
    if not args.skip_deps:
        if not args.frontend_only:
            check_python_dependencies()
        if not args.backend_only:
            if not check_node_dependencies():
                if not args.frontend_only:
                    print("Continuing with backend only.")
                    args.backend_only = True
                else:
                    print("Exiting due to frontend dependency issues.")
                    return

    processes = []
    try:
        if not args.frontend_only:
            backend_process = start_backend()
            processes.append(backend_process)
            time.sleep(2)  # Wait for backend to initialize

        if not args.backend_only:
            frontend_process = start_frontend()
            processes.append(frontend_process)

        print("\nAll services started successfully!")
        print("Backend API: http://localhost:8000")
        print("Frontend: http://localhost:3000")
        print("\nPress Ctrl+C to stop all services...")

        # Wait for processes to complete or user interrupt
        for process in processes:
            process.wait()

    except KeyboardInterrupt:
        print("\nShutting down services...")
        for process in processes:
            if platform.system() == "Windows":
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)
            process.wait()
        print("Services stopped successfully")

if __name__ == "__main__":
    main()