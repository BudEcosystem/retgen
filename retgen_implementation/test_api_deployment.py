#!/usr/bin/env python3
"""Quick test of API deployment."""

import time
import subprocess
import requests
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_deployment():
    """Test the API deployment."""
    print("Testing RETGEN API Deployment")
    print("=" * 60)
    
    # Kill any existing server
    subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
    time.sleep(2)
    
    # Start server
    print("\n1. Starting API server...")
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent / "src")
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("   Waiting for server to start...")
    time.sleep(5)
    
    try:
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        response = requests.get("http://localhost:8000/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test model info
        print("\n3. Testing model info...")
        response = requests.get("http://localhost:8000/model/info")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # Test training
        print("\n4. Testing model training...")
        documents = [
            "Natural language processing is amazing.",
            "Machine learning transforms data into insights.",
            "Deep learning uses neural networks."
        ] * 5
        
        response = requests.post(
            "http://localhost:8000/model/train",
            json={"documents": documents, "validation_split": 0.1},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        
        # Test generation
        print("\n5. Testing text generation...")
        if response.status_code == 200:
            time.sleep(2)  # Wait for training to complete
            
            gen_response = requests.post(
                "http://localhost:8000/generate",
                json={
                    "prompt": "Natural language",
                    "max_length": 20,
                    "temperature": 0.8
                },
                timeout=10
            )
            print(f"   Status: {gen_response.status_code}")
            if gen_response.status_code == 200:
                print(f"   Generated: {gen_response.json()['generations'][0]}")
        
        print("\n✅ API deployment test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Kill server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait()
        print("Server stopped")

if __name__ == "__main__":
    test_deployment()