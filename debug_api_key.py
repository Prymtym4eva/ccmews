#!/usr/bin/env python3
"""
Debug script to check why API key isn't being sent
"""
import json
import os
from pathlib import Path

print("=" * 60)
print("SMS CONFIG DEBUG")
print("=" * 60)

# Check config file
config_path = Path('sms_config.json')
print(f"\n1. Config file exists: {config_path.exists()}")

if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    
    frog = config.get('frog', {})
    api_key = frog.get('api_key', '')
    username = frog.get('username', '')
    
    print(f"\n2. Frog config section exists: {'frog' in config}")
    print(f"3. API Key from config:")
    print(f"   - Value: '{api_key}'")
    print(f"   - Length: {len(api_key) if api_key else 0}")
    print(f"   - Is empty: {not api_key}")
    print(f"   - Is None: {api_key is None}")
    print(f"   - Bool value: {bool(api_key)}")
    
    print(f"\n4. Username from config:")
    print(f"   - Value: '{username}'")
    print(f"   - Is empty: {not username}")
    
    print(f"\n5. Environment variables:")
    print(f"   - FROG_API_KEY: {os.environ.get('FROG_API_KEY', 'NOT SET')}")
    print(f"   - FROG_USERNAME: {os.environ.get('FROG_USERNAME', 'NOT SET')}")
    
    # Check what would actually be used
    final_api_key = frog.get("api_key") or os.environ.get("FROG_API_KEY")
    final_username = frog.get("username") or os.environ.get("FROG_USERNAME")
    
    print(f"\n6. Final values that would be used:")
    print(f"   - API Key: '{final_api_key[:20] if final_api_key else 'NONE'}...'")
    print(f"   - Username: '{final_username}'")
    
    # Test actual headers
    print(f"\n7. Headers that would be sent:")
    headers = {
        "Content-Type": "application/json",
        "API-KEY": final_api_key or "",
        "USERNAME": final_username or ""
    }
    for k, v in headers.items():
        print(f"   {k}: '{v[:30] if v else 'EMPTY'}{'...' if v and len(v) > 30 else ''}'")
    
    # Issue detection
    print(f"\n8. ISSUES DETECTED:")
    if not final_api_key:
        print("   ❌ API KEY IS EMPTY - This is the problem!")
    else:
        print("   ✅ API Key is set")
    
    if not final_username:
        print("   ❌ USERNAME IS EMPTY")
    else:
        print("   ✅ Username is set")

else:
    print("❌ sms_config.json not found!")

# Now test with direct values
print("\n" + "=" * 60)
print("DIRECT API TEST")
print("=" * 60)

import requests

API_KEY = "$2a$10$koPdxjSWFfIXGVQ0OiWPbOMuGxabVCC4pgwUoxpTOzjOyu1CBRNNW"
USERNAME = "benaikins"

headers = {
    "Content-Type": "application/json",
    "API-KEY": API_KEY,
    "USERNAME": USERNAME
}

payload = {
    "senderid": "ccmews",
    "destinations": [{"destination": "0556969806", "msgid": "DEBUG001"}],
    "message": "Debug test",
    "smstype": "text"
}

print(f"\nSending with hardcoded credentials...")
print(f"API-KEY header: {headers['API-KEY'][:30]}...")
print(f"USERNAME header: {headers['USERNAME']}")

try:
    response = requests.post(
        "https://frogapi.wigal.com.gh/api/v3/sms/send",
        headers=headers,
        json=payload,
        timeout=30
    )
    print(f"\nResponse: {response.status_code}")
    print(f"Body: {response.text}")
except Exception as e:
    print(f"Error: {e}")
