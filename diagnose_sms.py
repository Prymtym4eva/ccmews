#!/usr/bin/env python3
"""
Diagnose why ccmews_sms_alerts.py doesn't work but debug_sms.py does
"""
import json
from pathlib import Path

print("=" * 60)
print("SMS ALERTS DIAGNOSTIC")
print("=" * 60)

# Check config
config_path = Path('sms_config.json')
with open(config_path) as f:
    config = json.load(f)

print("\n--- CONFIG CHECK ---")
print(f"enabled: {config.get('enabled')} (must be True)")
print(f"test_mode: {config.get('test_mode')} (must be False to send real SMS)")
print(f"provider: {config.get('provider')}")

frog = config.get('frog', {})
print(f"\nFrog API Key: {frog.get('api_key', 'MISSING')[:30]}...")
print(f"Frog Username: {frog.get('username', 'MISSING')}")
print(f"Frog Sender ID: {frog.get('sender_id', 'MISSING')}")
print(f"Use Test API: {frog.get('use_test_api', 'not set')}")

recipients = config.get('recipients', [])
print(f"\nRecipients: {len(recipients)}")
for r in recipients:
    print(f"  - {r.get('name')}: {r.get('phone')} (active: {r.get('active')})")

print("\n--- TESTING SMS ALERTS MODULE ---")

try:
    from ccmews_sms_alerts import SMSConfig, AlertSystem, FrogProvider
    print("✅ Module imported")
    
    # Check what config the module loads
    sms_config = SMSConfig()
    print(f"\nModule sees enabled: {sms_config.config.get('enabled')}")
    print(f"Module sees test_mode: {sms_config.config.get('test_mode')}")
    print(f"Module sees provider: {sms_config.config.get('provider')}")
    
    # Try to create alert system
    print("\n--- CREATING ALERT SYSTEM ---")
    alert_system = AlertSystem()
    print(f"Provider type: {type(alert_system.provider).__name__}")
    
    if hasattr(alert_system.provider, 'api_key'):
        print(f"Provider API Key: {alert_system.provider.api_key[:30]}...")
    if hasattr(alert_system.provider, 'username'):
        print(f"Provider Username: {alert_system.provider.username}")
    if hasattr(alert_system.provider, 'sender_id'):
        print(f"Provider Sender ID: {alert_system.provider.sender_id}")
    if hasattr(alert_system.provider, 'api_url'):
        print(f"Provider API URL: {alert_system.provider.api_url}")
    
    # Try sending
    print("\n--- ATTEMPTING TO SEND ---")
    result = alert_system.send_flood_alert(
        location="Test Location",
        risk=0.55,
        rainfall_mm=35,
        when="Test - Debug"
    )
    print(f"Result: {result}")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    traceback.print_exc()
