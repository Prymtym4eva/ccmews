#!/usr/bin/env python3
"""
CCMEWS SMS Debug Script
Detailed debugging for Frog API issues
"""
import requests
import json
from datetime import datetime

print("=" * 60)
print("FROG SMS API DEBUG")
print("=" * 60)

# Load config
try:
    with open('sms_config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config loaded")
except Exception as e:
    print(f"❌ Failed to load config: {e}")
    exit(1)

# Extract credentials
frog_config = config.get('frog', {})
API_KEY = frog_config.get('api_key', '')
USERNAME = frog_config.get('username', '')
SENDER_ID = frog_config.get('sender_id', '')

recipients = config.get('recipients', [])
PHONE = recipients[0].get('phone', '') if recipients else ''

# Clean phone number
if PHONE.startswith('+233'):
    PHONE = '0' + PHONE[4:]
elif PHONE.startswith('233'):
    PHONE = '0' + PHONE[3:]

print()
print("--- CREDENTIALS ---")
print(f"API Key: {API_KEY[:20]}...{API_KEY[-10:]}" if len(API_KEY) > 30 else f"API Key: {API_KEY}")
print(f"Username: {USERNAME}")
print(f"Sender ID: {SENDER_ID}")
print(f"Phone: {PHONE}")
print()

# Prepare request
API_URL = "https://frogapi.wigal.com.gh/api/v3/sms/send"

headers = {
    "Content-Type": "application/json",
    "API-KEY": API_KEY,
    "USERNAME": USERNAME
}

payload = {
    "senderid": SENDER_ID,
    "destinations": [
        {
            "destination": PHONE,
            "msgid": f"TEST-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    ],
    "message": "CCMEWS Test Message - " + datetime.now().strftime('%H:%M:%S'),
    "smstype": "text"
}

print("--- REQUEST DETAILS ---")
print(f"URL: {API_URL}")
print(f"Headers: {json.dumps({k: v[:20]+'...' if len(v) > 20 else v for k,v in headers.items()}, indent=2)}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print()

print("--- SENDING REQUEST ---")
try:
    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=30
    )
    
    print(f"HTTP Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {response.text}")
    print()
    
    if response.status_code == 200:
        result = response.json()
        if result.get('status', '').upper() in ['ACCEPTD', 'SUCCESS', 'ACCEPTED']:
            print("✅ SUCCESS! Message sent!")
        else:
            print(f"⚠️ API Response: {result}")
    elif response.status_code == 403:
        print()
        print("❌ 403 FORBIDDEN - Possible causes:")
        print("   1. API Key is incorrect")
        print("   2. Username is incorrect") 
        print("   3. Sender ID not approved")
        print("   4. Account doesn't have API access")
        print("   5. API access not yet activated (check email)")
    elif response.status_code == 401:
        print()
        print("❌ 401 UNAUTHORIZED - Check credentials")
    elif response.status_code == 404:
        print()
        print("❌ 404 NOT FOUND - Sender ID might not exist")
    else:
        print(f"❌ Error: HTTP {response.status_code}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Request Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 60)
print("ALTERNATIVE TESTS TO TRY:")
print("=" * 60)
print()
print("1. Try with your username as sender ID:")
print(f'   Change "sender_id" to "{USERNAME}" in sms_config.json')
print()
print("2. Try the test API endpoint:")
print("   Change API_URL to: https://frogtestapi.wigal.com.gh/api/v3/sms/send")
print()
print("3. Check if API key has special characters that need escaping")
print()
print("4. Contact Frog support: https://sms.wigal.com.gh")
