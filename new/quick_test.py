#!/usr/bin/env python3
"""
Simple admin endpoint test
"""
import requests
import json

def main():
    """Test admin endpoints quickly"""
    print("🔍 Testing Admin Endpoints")
    print("=" * 30)
    
    base_url = "http://localhost:8000"
    
    # Test endpoints
    endpoints = [
        "/health",
        "/admin/dashboard", 
        "/admin/quality-report",
        "/admin/feedback-analysis"
    ]
    
    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            print(f"\n📡 Testing: {endpoint}")
            
            response = requests.get(url, timeout=5)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Keys: {list(data.keys())}")
                if 'system' in data:
                    print(f"      System entries: {data['system'].get('total_entries', 'N/A')}")
                if 'quality' in data:
                    print(f"      Avg quality: {data['quality'].get('average_score', 'N/A')}")
            else:
                print(f"   ❌ Failed: {response.text[:100]}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection failed")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    main()