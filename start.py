import subprocess
import sys
import time
from pathlib import Path
import argparse
import signal
import os

def start_backend():
    """Start the FastAPI backend server"""
    print("Starting backend server...")
    backend_path = Path("backend")
    env = os.environ.copy()
    
    if sys.platform == "win32":
        python = "python"
    else:
        python = "python3"
    
    return subprocess.Popen(
        [python, "src/main.py"],
        cwd=str(backend_path),
        env=env
    )

def start_frontend():
    """Start the Next.js frontend development server"""
    print("Starting frontend development server...")
    frontend_path = Path("frontend")
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_path)
    )

def main():
    parser = argparse.ArgumentParser(description="Start OCR system services")
    parser.add_argument("--backend-only", action="store_true", help="Start only the backend server")
    parser.add_argument("--frontend-only", action="store_true", help="Start only the frontend server")
    args = parser.parse_args()

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
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                process.terminate()
            process.wait()
        print("Services stopped successfully")

if __name__ == "__main__":
    main()