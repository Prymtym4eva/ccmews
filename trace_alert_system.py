#!/usr/bin/env python3
"""
Trace exactly what's happening in AlertSystem
"""
import json
from pathlib import Path

print("=" * 60)
print("ALERT SYSTEM TRACE")
print("=" * 60)

# Import and trace
from ccmews_sms_alerts import SMSConfig, AlertSystem, FrogProvider

print("\n1. Loading SMSConfig...")
config = SMSConfig()
print(f"   Config loaded from: {config.config_path}")

frog_config = config.config.get("frog", {})
print(f"\n2. Frog config from SMSConfig:")
print(f"   api_key: '{frog_config.get('api_key', 'MISSING')[:30]}...'")
print(f"   username: '{frog_config.get('username', 'MISSING')}'")
print(f"   sender_id: '{frog_config.get('sender_id', 'MISSING')}'")
print(f"   use_test_api: {frog_config.get('use_test_api')}")

print(f"\n3. test_mode setting: {config.config.get('test_mode')}")
print(f"   enabled setting: {config.config.get('enabled')}")

print("\n4. Creating AlertSystem...")
alert_system = AlertSystem()

print(f"\n5. Provider type: {type(alert_system.provider).__name__}")

if isinstance(alert_system.provider, FrogProvider):
    print(f"\n6. FrogProvider attributes:")
    print(f"   api_key: '{alert_system.provider.api_key[:30] if alert_system.provider.api_key else 'EMPTY'}...'")
    print(f"   username: '{alert_system.provider.username}'")
    print(f"   sender_id: '{alert_system.provider.sender_id}'")
    print(f"   api_url: '{alert_system.provider.api_url}'")
    
    # Check if api_key is actually being set
    if not alert_system.provider.api_key:
        print("\n   ❌ API KEY IS EMPTY IN PROVIDER!")
    else:
        print("\n   ✅ API key is set in provider")
        
    # Test send directly
    print("\n7. Testing direct send via provider...")
    result = alert_system.provider.send("0556969806", "Test from trace script")
    print(f"   Result: {result}")
else:
    print(f"\n6. Provider is NOT FrogProvider - it's {type(alert_system.provider).__name__}")
    print("   This means test_mode is True or there's a config issue")
