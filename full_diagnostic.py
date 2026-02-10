#!/usr/bin/env python3
"""
CCMEWS Full System Diagnostic
Tests all components: data fetching, AI predictions, and SMS alerts
"""
import sys
import traceback
from pathlib import Path

print("=" * 70)
print("CCMEWS FULL SYSTEM DIAGNOSTIC")
print("=" * 70)

errors = []

# Test 1: Check all required files exist
print("\n📁 TEST 1: Required Files")
print("-" * 50)

required_files = [
    "ccmews_data_service.py",
    "ccmews_ai_engine.py", 
    "ccmews_sms_alerts.py",
    "ccmews_scheduler_service.py",
    "ccmews_event_forecast.py",
    "sms_config.json"
]

for f in required_files:
    exists = Path(f).exists()
    status = "✅" if exists else "❌ MISSING"
    print(f"   {f}: {status}")
    if not exists:
        errors.append(f"Missing file: {f}")

# Test 2: Import all modules
print("\n📦 TEST 2: Module Imports")
print("-" * 50)

modules_ok = True

try:
    from ccmews_data_service import ClimateDataService, MONITORING_GRID
    print("   ✅ ccmews_data_service imported")
except Exception as e:
    print(f"   ❌ ccmews_data_service FAILED: {e}")
    errors.append(f"Import ccmews_data_service: {e}")
    modules_ok = False

try:
    from ccmews_ai_engine import run_prediction_cycle
    print("   ✅ ccmews_ai_engine imported")
except Exception as e:
    print(f"   ❌ ccmews_ai_engine FAILED: {e}")
    errors.append(f"Import ccmews_ai_engine: {e}")
    modules_ok = False

try:
    from ccmews_sms_alerts import AlertSystem, SMSConfig
    print("   ✅ ccmews_sms_alerts imported")
except Exception as e:
    print(f"   ❌ ccmews_sms_alerts FAILED: {e}")
    errors.append(f"Import ccmews_sms_alerts: {e}")
    modules_ok = False

try:
    from ccmews_event_forecast import EventForecaster
    print("   ✅ ccmews_event_forecast imported")
except Exception as e:
    print(f"   ❌ ccmews_event_forecast FAILED: {e}")
    errors.append(f"Import ccmews_event_forecast: {e}")
    modules_ok = False

# Test 3: Data Fetching
print("\n📡 TEST 3: Data Fetching (Open-Meteo API)")
print("-" * 50)

if modules_ok:
    try:
        data_service = ClimateDataService()
        
        # Test with single location first
        print("   Testing single location fetch (Battor)...")
        
        # Check if fetch method exists
        if hasattr(data_service, 'fetch_current_weather'):
            result = data_service.fetch_current_weather(6.0833, 0.4200)
            if result:
                print(f"   ✅ Single location fetch: SUCCESS")
                print(f"      Temperature: {result.get('temperature_2m', 'N/A')}°C")
                print(f"      Precipitation: {result.get('precipitation', 'N/A')}mm")
            else:
                print("   ⚠️ Single location fetch returned empty")
        elif hasattr(data_service, 'fetch_all_locations'):
            print("   Fetching all locations (may take 30-60 seconds)...")
            locations, records = data_service.fetch_all_locations()
            print(f"   ✅ Data fetch: {locations} locations, {records} records")
        else:
            print("   ⚠️ No fetch method found - checking available methods:")
            methods = [m for m in dir(data_service) if not m.startswith('_')]
            print(f"      Available methods: {methods}")
            
    except Exception as e:
        print(f"   ❌ Data fetch FAILED: {e}")
        traceback.print_exc()
        errors.append(f"Data fetch: {e}")
else:
    print("   ⏭️ Skipped (module import failed)")

# Test 4: AI Predictions
print("\n🧠 TEST 4: AI Predictions")
print("-" * 50)

if modules_ok:
    try:
        print("   Running prediction cycle...")
        
        # Use subset of monitoring grid for faster test
        test_grid = {
            "Battor": {"lat": 6.0833, "lon": 0.4200},
            "Adidome": {"lat": 6.1200, "lon": 0.3150},
        }
        
        prediction_result = run_prediction_cycle(test_grid)
        
        if prediction_result:
            predictions = prediction_result.get("predictions", [])
            print(f"   ✅ Predictions generated: {len(predictions)} locations")
            
            if predictions:
                sample = predictions[0]
                print(f"   Sample prediction for {sample.get('location', 'Unknown')}:")
                print(f"      Flood risk: {sample.get('flood_risk', 0):.1%}")
                print(f"      Heat risk: {sample.get('heat_risk', 0):.1%}")
                print(f"      Drought risk: {sample.get('drought_risk', 0):.1%}")
        else:
            print("   ⚠️ Prediction returned empty result")
            
    except Exception as e:
        print(f"   ❌ AI predictions FAILED: {e}")
        traceback.print_exc()
        errors.append(f"AI predictions: {e}")
else:
    print("   ⏭️ Skipped (module import failed)")

# Test 5: SMS Configuration
print("\n📲 TEST 5: SMS Configuration")
print("-" * 50)

if modules_ok:
    try:
        config = SMSConfig()
        frog = config.config.get("frog", {})
        
        print(f"   Provider: {config.config.get('provider', 'not set')}")
        print(f"   Enabled: {config.config.get('enabled', False)}")
        print(f"   Test Mode: {config.config.get('test_mode', True)}")
        print(f"   API Key: {'✅ Set' if frog.get('api_key') else '❌ MISSING'}")
        print(f"   Username: {'✅ Set' if frog.get('username') else '❌ MISSING'}")
        
        recipients = config.get_recipients()
        print(f"   Recipients: {len(recipients)}")
        for r in recipients:
            print(f"      - {r.name}: {r.phone}")
            
        if not frog.get('api_key'):
            errors.append("SMS: API key missing")
        if not recipients:
            errors.append("SMS: No recipients configured")
            
    except Exception as e:
        print(f"   ❌ SMS config FAILED: {e}")
        errors.append(f"SMS config: {e}")

# Test 6: SMS Provider
print("\n📤 TEST 6: SMS Provider Connection")
print("-" * 50)

if modules_ok:
    try:
        alert_system = AlertSystem()
        provider_type = type(alert_system.provider).__name__
        
        print(f"   Provider type: {provider_type}")
        
        if provider_type == "FrogProvider":
            print(f"   API URL: {alert_system.provider.api_url}")
            print(f"   Sender ID: {alert_system.provider.sender_id}")
            print(f"   API Key set: {'✅ Yes' if alert_system.provider.api_key else '❌ No'}")
            
            # Actually test sending
            print("\n   Sending test SMS...")
            result = alert_system.provider.send("0556969806", "CCMEWS Diagnostic Test")
            
            if result.get("success"):
                print(f"   ✅ SMS sent successfully!")
                print(f"      Message ID: {result.get('message_id')}")
            else:
                print(f"   ❌ SMS failed: {result.get('error')}")
                errors.append(f"SMS send: {result.get('error')}")
        elif provider_type == "TestProvider":
            print("   ⚠️ Using TestProvider (test_mode=true, no real SMS sent)")
            errors.append("SMS: test_mode is True, not sending real SMS")
        else:
            print(f"   ⚠️ Unknown provider type")
            
    except Exception as e:
        print(f"   ❌ SMS provider FAILED: {e}")
        traceback.print_exc()
        errors.append(f"SMS provider: {e}")

# Test 7: Event Forecaster
print("\n🌤️ TEST 7: Event Forecaster")
print("-" * 50)

if modules_ok:
    try:
        forecaster = EventForecaster()
        
        print("   Fetching forecast for Battor...")
        forecast = forecaster.get_full_forecast(6.0833, 0.4200)
        
        if forecast:
            rain = forecast.get("next_rainfall")
            heat = forecast.get("next_heat_event")
            
            if rain:
                print(f"   ✅ Next rainfall: {rain.get('start_time', 'N/A')}")
                print(f"      Amount: {rain.get('total_mm', 0):.1f}mm")
                print(f"      Intensity: {rain.get('intensity', 'N/A')}")
            else:
                print("   ℹ️ No rainfall forecast in next 7 days")
                
            if heat:
                print(f"   ✅ Next heat event: {heat.get('start_date', 'N/A')}")
                print(f"      Max temp: {heat.get('max_temperature', 0):.1f}°C")
            else:
                print("   ℹ️ No heat event forecast in next 7 days")
        else:
            print("   ⚠️ Forecast returned empty")
            
    except Exception as e:
        print(f"   ❌ Event forecaster FAILED: {e}")
        traceback.print_exc()
        errors.append(f"Event forecaster: {e}")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if not errors:
    print("\n✅ ALL TESTS PASSED - System is fully operational!")
    print("\nYou can now run:")
    print("   python ccmews_scheduler_service.py --run-once")
else:
    print(f"\n❌ {len(errors)} ISSUE(S) FOUND:\n")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    
    print("\n📋 RECOMMENDATIONS:")
    if any("Missing file" in e for e in errors):
        print("   - Download missing files from the outputs I provided")
    if any("API key" in e.lower() for e in errors):
        print("   - Check sms_config.json has correct API key")
    if any("test_mode" in e.lower() for e in errors):
        print("   - Set 'test_mode': false in sms_config.json")
    if any("Import" in e for e in errors):
        print("   - Install missing dependencies: pip install httpx pandas numpy scikit-learn")
