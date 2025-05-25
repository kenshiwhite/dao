import subprocess
import sys
import time
import requests
import threading
from pathlib import Path


def check_server_status(url, timeout=5):
    """Check if a server is running at the given URL"""
    try:
        response = requests.get(url, timeout=timeout)
        return True
    except requests.exceptions.RequestException:
        return False


def start_fastapi_server():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    try:
        # Check directory structure and adjust import path
        if Path("app/app.py").exists():
            # Structure: app/app.py
            cmd = [sys.executable, "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
            print("📁 Detected app/app.py structure")
        elif Path("app.py").exists():
            # Structure: app.py
            cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
            print("📁 Detected app.py structure")
        else:
            print("❌ Could not find app.py or app/app.py")
            return None

        process = subprocess.Popen(cmd, cwd=Path.cwd())

        # Wait for server to start
        print("⏳ Waiting for FastAPI server to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_server_status("http://localhost:8000/docs"):
                print("✅ FastAPI server is running at http://localhost:8000")
                print("📚 API documentation available at http://localhost:8000/docs")
                return process
            time.sleep(1)

        print("⚠️  FastAPI server might not have started properly")
        return process

    except Exception as e:
        print(f"❌ Failed to start FastAPI server: {e}")
        return None


def start_gradio_interface():
    """Start the Gradio frontend"""
    print("🚀 Starting Gradio frontend...")
    try:
        # Check for different possible locations of gradio file
        if Path("utils/gradio_temp.py").exists():
            cmd = [sys.executable, "utils/gradio_temp.py"]
            print("📁 Found gradio_temp.py in utils/")
        elif Path("gradio_temp.py").exists():
            cmd = [sys.executable, "gradio_temp.py"]
            print("📁 Found gradio_temp.py in root")
        else:
            print("❌ Could not find gradio_temp.py")
            return None

        process = subprocess.Popen(cmd, cwd=Path.cwd())
        print("✅ Gradio interface should be starting...")
        print("🌐 Interface will be available at http://localhost:7860")
        return process
    except Exception as e:
        print(f"❌ Failed to start Gradio interface: {e}")
        return None


def main():
    """Main function to start both servers"""
    print("🔧 CLIP Image Search Application Startup")
    print("=" * 50)

    # Check if required files exist
    if Path("app/app.py").exists():
        required_files = ["app/app.py"]
        if Path("utils/gradio_temp.py").exists():
            required_files.append("utils/gradio_temp.py")
        elif Path("gradio_temp.py").exists():
            required_files.append("gradio_temp.py")
    else:
        required_files = ["app.py", "gradio_temp.py"]

    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all files are in the current directory.")
        return

    try:
        # Start FastAPI backend
        fastapi_process = start_fastapi_server()
        if not fastapi_process:
            return

        # Give FastAPI a moment to fully initialize
        time.sleep(3)

        # Start Gradio frontend
        gradio_process = start_gradio_interface()
        if not gradio_process:
            fastapi_process.terminate()
            return

        print("\n" + "=" * 50)
        print("🎉 Both servers are starting!")
        print("📊 FastAPI Backend: http://localhost:8000")
        print("🖥️  Gradio Frontend: http://localhost:7860")
        print("📚 API Docs: http://localhost:8000/docs")
        print("=" * 50)
        print("\n⌨️  Press Ctrl+C to stop both servers")

        # Wait for user interrupt
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")

    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
    finally:
        # Clean up processes
        if 'fastapi_process' in locals() and fastapi_process:
            fastapi_process.terminate()
            print("✅ FastAPI server stopped")
        if 'gradio_process' in locals() and gradio_process:
            gradio_process.terminate()
            print("✅ Gradio interface stopped")


if __name__ == "__main__":
    main()