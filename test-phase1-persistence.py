#!/usr/bin/env python3
"""
Test script for Phase 1 tasks 1.5 and 1.6:
- Persistent job storage 
- Improved error messages
"""

import requests
import time
import json

BASE_URL = "http://localhost:8005"

def test_improved_errors():
    """Test improved error messages for task 1.6."""
    print("=== Testing Improved Error Messages (Task 1.6) ===")
    
    # Test chat with non-existent model
    try:
        response = requests.post(f"{BASE_URL}/chat", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hello"}]
        })
        print(f"❌ Expected error but got: {response.status_code}")
    except requests.exceptions.RequestException:
        print("✅ Server not running - start server first")
        return
    
    if response.status_code == 400:
        print(f"✅ Got 400 error with improved message: {response.json()['detail']}")
    else:
        print(f"⚠️  Got {response.status_code}: {response.text}")

def test_persistent_jobs():
    """Test persistent job storage for task 1.5."""
    print("\\n=== Testing Persistent Job Storage (Task 1.5) ===")
    
    # Start a model pull
    try:
        response = requests.post(f"{BASE_URL}/pull", json={
            "model": "gpt2",
            "init": False
        })
        
        if response.status_code == 202:
            job_id = response.json()["job_id"]
            print(f"✅ Started pull job: {job_id}")
            
            # Check job status
            job_response = requests.get(f"{BASE_URL}/jobs/{job_id}")
            if job_response.status_code == 200:
                job_data = job_response.json()
                print(f"✅ Job found in database: {job_data['status']}")
                return job_id
            else:
                print(f"❌ Job not found: {job_response.status_code}")
        else:
            print(f"❌ Pull failed: {response.status_code} {response.text}")
            
    except requests.exceptions.RequestException:
        print("❌ Server not running - start server first")
        return None

def test_job_persistence_after_restart():
    """Test that jobs persist after server restart."""
    print("\\n=== Testing Job Persistence (requires manual restart) ===")
    print("1. Start server: python -m uvicorn app:app --host 0.0.0.0 --port 8005")
    print("2. Run this test to create a job")  
    print("3. Stop server (Ctrl+C)")
    print("4. Restart server")
    print("5. Check if job still exists in database")

def main():
    print("Phase 1 Tasks 1.5 & 1.6 Test Suite")
    print("===================================")
    
    # Test error messages first
    test_improved_errors()
    
    # Test persistent jobs
    job_id = test_persistent_jobs()
    
    # Print instructions for restart testing
    test_job_persistence_after_restart()
    
    if job_id:
        print(f"\\n💡 Job ID for restart testing: {job_id}")
        print(f"   Check with: GET {BASE_URL}/jobs/{job_id}")

if __name__ == "__main__":
    main()