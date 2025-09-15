#!/usr/bin/env python3
"""
Test script for admin endpoints
"""
import requests
import json
import time

def test_endpoint(url, name):
    """Test a single endpoint"""
    try:
        print(f"\n🔍 Testing {name}...")
        print(f"URL: {url}")
        
        response = requests.get(url, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(f"Response preview: {json.dumps(data, indent=2)[:500]}...")
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - server not running")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Test all admin endpoints"""
    print("🚀 Testing SAT Vocab RAG Admin Endpoints")
    print("=" * 50)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    time.sleep(20)
    
    base_url = "http://localhost:8000"
    
    # Test basic health check first
    test_endpoint(f"{base_url}/health", "Health Check")
    
    # Test admin endpoints
    admin_endpoints = [
        ("admin/dashboard", "Admin Dashboard"),
        ("admin/quality-report", "Quality Report"),  
        ("admin/feedback-analysis", "Feedback Analysis")
    ]
    
    for endpoint, name in admin_endpoints:
        test_endpoint(f"{base_url}/{endpoint}", name)
    
    print("\n✨ Testing complete!")

if __name__ == "__main__":
    main()