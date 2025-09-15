#!/usr/bin/env python3
"""
Detailed admin endpoint response test
"""
import requests
import json
from subprocess import Popen
import time
import os

def pretty_print_response(endpoint, response_data):
    """Pretty print response data"""
    print(f"\n{'='*60}")
    print(f"📊 {endpoint.upper()} RESPONSE")
    print(f"{'='*60}")
    print(json.dumps(response_data, indent=2))

def main():
    """Test admin endpoints in detail"""
    print("🚀 Starting detailed admin endpoint test...")
    
    # Start server
    server = Popen(['python', 'start.py'], cwd=os.getcwd())
    print("⏳ Waiting 25 seconds for server startup...")
    time.sleep(25)
    
    try:
        base_url = 'http://localhost:8000'
        
        # Test each admin endpoint
        endpoints = [
            '/api/v1/admin/dashboard',
            '/api/v1/admin/quality-report', 
            '/api/v1/admin/feedback-analysis'
        ]
        
        for endpoint in endpoints:
            try:
                url = f'{base_url}{endpoint}'
                print(f"\n🔍 Fetching: {endpoint}")
                
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    pretty_print_response(endpoint.split('/')[-1], data)
                else:
                    print(f"❌ Failed: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"❌ Error testing {endpoint}: {e}")
        
    finally:
        # Clean up
        print(f"\n{'='*60}")
        print("🛑 Stopping server...")
        server.terminate()
        server.wait()
        print("✅ Test complete!")

if __name__ == "__main__":
    main()