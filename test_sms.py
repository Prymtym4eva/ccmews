#!/usr/bin/env python3
"""
CCMEWS SMS Test Script
Run this to test your Frog by Wigal SMS configuration
"""
import requests
import json
from datetime import datetime

# =============================================================
# CONFIGURATION - Update these with your credentials
# =============================================================
API_KEY = "$2a$10$koPdxjSWFfIXGVQ0OiWPbOMuGxabVCC4pgwUoxpTOzjOyu1CBRNNW"
USERNAME = "benaikins"
SENDER_ID = "ccmews"
TEST_PHONE = "0556969806"  # Your phone number (0XX format)

# API Endpoints
PRODUCTION_API = "https://frogapi.wigal.com.gh/api/v3/sms/send"
TEST_API = "https://frogtestapi.wigal.com.gh/api/v3/sms/send"

# =============================================================
# TEST FUNCTIONS
# =============================================================

def send_test_sms(use_test_api=False):
    """Send a test SMS"""
    
    api_url = TEST_API if use_test_api else PRODUCTION_API
    
    # Test message
    message = f"""🔔 CCMEWS TEST ALERT

This is a test message from the Climate Change Monitoring & Early Warning System (CCMEWS).

If you received this, SMS alerts are working correctly!

Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
API Mode: {'TEST' if use_test_api else 'PRODUCTION'}"""

    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "API-KEY": API_KEY,
        "USERNAME": USERNAME
    }
    
    msg_id = f"CCMEWS-TEST-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    payload = {
        "senderid": SENDER_ID,
        "destinations": [
            {
                "destination": TEST_PHONE,
                "msgid": msg_id
            }
        ],
        "message": message,
        "smstype": "text"
    }
    
    print("=" * 50)
    print("CCMEWS SMS TEST")
    print("=" * 50)
    print(f"API URL: {api_url}")
    print(f"Username: {USERNAME}")
    print(f"Sender ID: {SENDER_ID}")
    print(f"Recipient: {TEST_PHONE}")
    print(f"Message ID: {msg_id}")
    print("-" * 50)
    print("Message:")
    print(message)
    print("-" * 50)
    print()
    print("📤 Sending SMS...")
    print()
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"HTTP Status: {response.status_code}")
        print(f"Response: {response.text}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "").upper()
            
            if status in ["SUCCESS", "ACCEPTED"]:
                print("✅ SUCCESS! SMS sent successfully!")
                print()
                print("Check your phone for the message.")
                return True
            else:
                print(f"⚠️ API returned status: {status}")
                print(f"Message: {result.get('message', 'No message')}")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: Could not connect to Frog API")
        print(f"   Check your internet connection")
        print(f"   Error: {e}")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout: Request took too long")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_balance():
    """Check SMS balance (if supported)"""
    print("\n📊 Checking account balance...")
    
    headers = {
        "Content-Type": "application/json",
        "API-KEY": API_KEY,
        "USERNAME": USERNAME
    }
    
    try:
        # Try balance endpoint
        response = requests.get(
            "https://frogapi.wigal.com.gh/api/v3/sms/balance",
            headers=headers,
            timeout=30
        )
        print(f"Balance Response: {response.text}")
    except Exception as e:
        print(f"Could not check balance: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CCMEWS SMS")
    parser.add_argument("--test-api", action="store_true", 
                       help="Use Frog test API (no real SMS sent)")
    parser.add_argument("--balance", action="store_true",
                       help="Check SMS balance")
    parser.add_argument("--phone", type=str,
                       help="Override test phone number")
    
    args = parser.parse_args()
    
    if args.phone:
        TEST_PHONE = args.phone
    
    if args.balance:
        check_balance()
    else:
        success = send_test_sms(use_test_api=args.test_api)
        
        if not success:
            print()
            print("=" * 50)
            print("TROUBLESHOOTING TIPS:")
            print("=" * 50)
            print("1. Check your API key is correct")
            print("2. Check your username is correct")
            print("3. Ensure sender ID is registered with Frog")
            print("4. Verify phone number format (0XXXXXXXXX)")
            print("5. Check you have SMS credits in your account")
            print("6. Try the test API first: --test-api")
            print()
            print("Frog Dashboard: https://sms.wigal.com.gh")
