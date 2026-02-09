"""
CCMEWS SMS Alert System
Sends SMS notifications to officials when hazards are forecasted

Supports:
1. Frog by Wigal (RECOMMENDED for Ghana):
   - Sign up at https://sms.wigal.com.gh
   - Get API key and username from dashboard
   - Best rates for Ghana networks

2. Africa's Talking (alternative):
   - Sign up at https://africastalking.com
   - Get API key and username

3. Twilio (international):
   - Sign up at https://twilio.com
   - Get Account SID, Auth Token, and phone number
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import sqlite3
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CCMEWS-SMS')

# Paths
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "sms_config.json"
ALERT_LOG_PATH = BASE_DIR / "alert_log.db"

# Try to import required libraries
REQUESTS_AVAILABLE = False
AFRICASTALKING_AVAILABLE = False
TWILIO_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass

try:
    import africastalking
    AFRICASTALKING_AVAILABLE = True
except ImportError:
    pass

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    pass


@dataclass
class AlertRecipient:
    """SMS alert recipient"""
    name: str
    phone: str  # International format: +233XXXXXXXXX
    role: str   # e.g., "District Emergency Coordinator", "NADMO Officer"
    district: str
    active: bool = True
    
    def to_dict(self):
        return asdict(self)


@dataclass
class AlertMessage:
    """SMS alert message"""
    alert_type: str  # 'flood', 'heat', 'drought', 'rainfall', 'combined'
    severity: str    # 'advisory', 'watch', 'warning', 'emergency'
    location: str
    message: str
    timestamp: str
    data: Dict
    
    def to_dict(self):
        return asdict(self)


class SMSConfig:
    """SMS configuration manager"""
    
    DEFAULT_CONFIG = {
        "provider": "frog",  # "frog" (recommended), "africastalking", or "twilio"
        "frog": {
            "api_key": "",      # Your Frog API key from dashboard
            "username": "",     # Your Frog username
            "sender_id": "CCMEWS",  # Sender ID (max 11 chars, must be registered)
            "use_test_api": True,   # Use test API for development
        },
        "africastalking": {
            "username": "",  # Your Africa's Talking username
            "api_key": "",   # Your Africa's Talking API key
            "sender_id": "CCMEWS",  # Sender ID (max 11 chars)
        },
        "twilio": {
            "account_sid": "",
            "auth_token": "",
            "from_number": "",  # Your Twilio phone number
        },
        "alert_thresholds": {
            "flood_risk": 0.45,      # Send alert if risk >= this
            "heat_risk": 0.45,
            "drought_risk": 0.45,
            "composite_risk": 0.50,
            "rainfall_mm": 30,       # Alert for rainfall >= this
            "temperature_c": 37,     # Alert for temp >= this
        },
        "recipients": [],
        "enabled": False,  # Set to True to enable SMS sending
        "test_mode": True,  # If True, logs messages instead of sending
    }
    
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in self.DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {self.config_path}")
    
    def add_recipient(self, name: str, phone: str, role: str, district: str):
        """Add an alert recipient"""
        recipient = AlertRecipient(name, phone, role, district)
        self.config["recipients"].append(recipient.to_dict())
        self.save_config()
        logger.info(f"Added recipient: {name} ({phone})")
    
    def remove_recipient(self, phone: str):
        """Remove a recipient by phone number"""
        self.config["recipients"] = [
            r for r in self.config["recipients"] if r["phone"] != phone
        ]
        self.save_config()
    
    def get_recipients(self, district: str = None) -> List[AlertRecipient]:
        """Get active recipients, optionally filtered by district
        
        If district is specified but no exact matches found, 
        returns all active recipients (for emergency alerts)
        """
        recipients = []
        all_active = []
        
        for r in self.config["recipients"]:
            if r.get("active", True):
                recipient = AlertRecipient(**r)
                all_active.append(recipient)
                
                # Check for district match (flexible matching)
                if district is None:
                    recipients.append(recipient)
                elif r.get("district") == district:
                    recipients.append(recipient)
                elif district and r.get("district") and (
                    district.lower() in r.get("district", "").lower() or
                    r.get("district", "").lower() in district.lower()
                ):
                    # Partial match (e.g., "Battor, North Tongu" matches "North Tongu")
                    recipients.append(recipient)
        
        # If no matches but we have active recipients, send to all (emergency fallback)
        if not recipients and all_active:
            return all_active
            
        return recipients


class AlertLogger:
    """Logs all sent alerts to SQLite database"""
    
    def __init__(self, db_path: Path = ALERT_LOG_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the alert log database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                location TEXT NOT NULL,
                recipient_name TEXT,
                recipient_phone TEXT,
                message TEXT NOT NULL,
                status TEXT NOT NULL,
                provider TEXT,
                message_id TEXT,
                error TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_alert(self, alert: AlertMessage, recipient: AlertRecipient,
                  status: str, provider: str = None, message_id: str = None, 
                  error: str = None):
        """Log an alert to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alert_log 
            (timestamp, alert_type, severity, location, recipient_name, 
             recipient_phone, message, status, provider, message_id, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.timestamp, alert.alert_type, alert.severity, alert.location,
            recipient.name, recipient.phone, alert.message, status,
            provider, message_id, error
        ))
        conn.commit()
        conn.close()
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts from the log"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM alert_log 
            WHERE datetime(timestamp) >= datetime('now', ?)
            ORDER BY timestamp DESC
        ''', (f'-{hours} hours',))
        
        columns = [desc[0] for desc in cursor.description]
        alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return alerts


class SMSProvider:
    """Base SMS provider interface"""
    
    def send(self, phone: str, message: str) -> Dict:
        raise NotImplementedError


class FrogProvider(SMSProvider):
    """
    Frog by Wigal SMS provider (Recommended for Ghana)
    API Documentation: https://frogdocs.wigal.com.gh/
    """
    
    # API endpoints
    PRODUCTION_URL = "https://frogapi.wigal.com.gh/api/v3/sms/send"
    TEST_URL = "https://frogtestapi.wigal.com.gh/api/v3/sms/send"
    
    def __init__(self, api_key: str, username: str, sender_id: str = "CCMEWS", use_test_api: bool = False):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not installed. Run: pip install requests")
        
        self.api_key = api_key
        self.username = username
        self.sender_id = sender_id
        self.api_url = self.TEST_URL if use_test_api else self.PRODUCTION_URL
        
        logger.info(f"Frog SMS initialized ({'TEST' if use_test_api else 'PRODUCTION'} mode)")
    
    def send(self, phone: str, message: str) -> Dict:
        """
        Send SMS via Frog by Wigal API
        
        Args:
            phone: Recipient phone number (can be 0XXXXXXXXX or 233XXXXXXXXX)
            message: Message content
            
        Returns:
            Dict with success status and message details
        """
        # Generate unique message ID
        msg_id = f"CCMEWS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Format phone number (Frog accepts both 0XX and 233XX formats)
        formatted_phone = phone.replace("+", "").replace(" ", "")
        if formatted_phone.startswith("233"):
            formatted_phone = "0" + formatted_phone[3:]
        
        # Clean message - replace special characters that might cause issues
        clean_message = message
        # Replace emojis with text equivalents
        emoji_replacements = {
            "⚠️": "[ALERT]",
            "🔥": "[HEAT]",
            "🌊": "[FLOOD]",
            "🏜️": "[DROUGHT]",
            "🌧️": "[RAIN]",
            "📲": "[SMS]",
            "🔔": "[ALERT]",
            "°C": " deg C",
            "°": " deg",
        }
        for emoji, replacement in emoji_replacements.items():
            clean_message = clean_message.replace(emoji, replacement)
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "API-KEY": self.api_key,
            "USERNAME": self.username
        }
        
        payload = {
            "senderid": self.sender_id,
            "destinations": [
                {
                    "destination": formatted_phone,
                    "msgid": msg_id
                }
            ],
            "message": clean_message,
            "smstype": "text"
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Try to parse JSON response
            try:
                result = response.json()
            except Exception as json_err:
                # If response isn't JSON, return the raw text
                return {
                    "success": False,
                    "message_id": msg_id,
                    "status": f"HTTP_{response.status_code}",
                    "error": f"Non-JSON response: {response.text[:200]}"
                }
            
            if response.status_code == 200:
                # Check response status
                status = result.get("status", "").upper()
                if status in ["SUCCESS", "ACCEPTED", "ACCEPTD"]:
                    return {
                        "success": True,
                        "message_id": msg_id,
                        "status": status,
                        "response": result,
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "message_id": msg_id,
                        "status": status,
                        "error": result.get("message", "Unknown error")
                    }
            else:
                return {
                    "success": False,
                    "message_id": msg_id,
                    "status": f"HTTP_{response.status_code}",
                    "error": result.get("message", f"HTTP error {response.status_code}")
                }
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timeout", "message_id": msg_id}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e), "message_id": msg_id}
        except Exception as e:
            return {"success": False, "error": str(e), "message_id": msg_id}
    
    def send_bulk(self, recipients: List[Dict], message: str) -> Dict:
        """
        Send SMS to multiple recipients in one API call
        
        Args:
            recipients: List of {"phone": "...", "name": "..."} dicts
            message: Message content
            
        Returns:
            Dict with bulk send results
        """
        msg_id_base = f"CCMEWS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        destinations = []
        for i, recipient in enumerate(recipients):
            phone = recipient.get("phone", "").replace("+", "").replace(" ", "")
            if phone.startswith("233"):
                phone = "0" + phone[3:]
            
            destinations.append({
                "destination": phone,
                "msgid": f"{msg_id_base}-{i:03d}"
            })
        
        headers = {
            "Content-Type": "application/json",
            "API-KEY": self.api_key,
            "USERNAME": self.username
        }
        
        payload = {
            "senderid": self.sender_id,
            "destinations": destinations,
            "message": message,
            "smstype": "text"
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            result = response.json()
            
            return {
                "success": response.status_code == 200,
                "total_recipients": len(destinations),
                "response": result,
                "error": None if response.status_code == 200 else result.get("message")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "total_recipients": len(destinations)}


class AfricasTalkingProvider(SMSProvider):
    """Africa's Talking SMS provider"""
    
    def __init__(self, username: str, api_key: str, sender_id: str = "CCMEWS"):
        if not AFRICASTALKING_AVAILABLE:
            raise ImportError("africastalking library not installed. Run: pip install africastalking")
        
        africastalking.initialize(username, api_key)
        self.sms = africastalking.SMS
        self.sender_id = sender_id
    
    def send(self, phone: str, message: str) -> Dict:
        """Send SMS via Africa's Talking"""
        try:
            response = self.sms.send(message, [phone], self.sender_id)
            
            # Parse response
            recipients = response.get("SMSMessageData", {}).get("Recipients", [])
            if recipients:
                recipient = recipients[0]
                return {
                    "success": recipient.get("status") == "Success",
                    "message_id": recipient.get("messageId"),
                    "status": recipient.get("status"),
                    "cost": recipient.get("cost"),
                    "error": None if recipient.get("status") == "Success" else recipient.get("status")
                }
            return {"success": False, "error": "No recipients in response"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class TwilioProvider(SMSProvider):
    """Twilio SMS provider"""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        if not TWILIO_AVAILABLE:
            raise ImportError("twilio library not installed. Run: pip install twilio")
        
        self.client = TwilioClient(account_sid, auth_token)
        self.from_number = from_number
    
    def send(self, phone: str, message: str) -> Dict:
        """Send SMS via Twilio"""
        try:
            msg = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=phone
            )
            return {
                "success": msg.status in ["queued", "sent", "delivered"],
                "message_id": msg.sid,
                "status": msg.status,
                "error": None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class TestProvider(SMSProvider):
    """Test provider that logs messages instead of sending"""
    
    def send(self, phone: str, message: str) -> Dict:
        logger.info(f"[TEST MODE] SMS to {phone}:\n{message}\n")
        return {
            "success": True,
            "message_id": f"TEST-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "test",
            "error": None
        }


class AlertSystem:
    """Main SMS alert system"""
    
    # Alert message templates (SMS-friendly, no emojis)
    TEMPLATES = {
        "flood_warning": """CCMEWS FLOOD ALERT
Location: {location}
Risk Level: {risk_level} ({risk_pct}%)
Expected Rain: {rainfall_mm}mm
When: {when}

ACTION: Monitor water levels and prepare evacuation routes.

Sent: {timestamp}""",

        "heat_warning": """CCMEWS HEAT ALERT
Location: {location}
Risk Level: {risk_level}
Max Temp: {max_temp} C
Feels Like: {heat_index} C
Duration: {duration}

ACTION: Advise vulnerable populations to stay indoors and hydrate.

Sent: {timestamp}""",

        "drought_warning": """CCMEWS DROUGHT ALERT
Location: {location}
Risk Level: {risk_level}
Dry Days: {dry_days}
Soil Moisture: {soil_moisture}

ACTION: Implement water conservation measures.

Sent: {timestamp}""",

        "rainfall_forecast": """CCMEWS RAINFALL FORECAST
Location: {location}
Expected: {intensity} rain
Amount: {rainfall_mm}mm
Starting: {start_time}
Duration: {duration}hrs
Probability: {probability}%

Sent: {timestamp}""",

        "combined_alert": """CCMEWS HAZARD ALERT
Location: {location}

{alerts}

Overall Risk: {composite_risk}%
ACTION REQUIRED

Sent: {timestamp}"""
    }
    
    def __init__(self, config: SMSConfig = None):
        self.config = config or SMSConfig()
        self.logger = AlertLogger()
        self.provider = self._init_provider()
    
    def _init_provider(self) -> SMSProvider:
        """Initialize the SMS provider based on config"""
        if self.config.config.get("test_mode", True):
            logger.info("Running in TEST MODE - messages will be logged, not sent")
            return TestProvider()
        
        provider_name = self.config.config.get("provider", "frog")
        
        if provider_name == "frog":
            frog_config = self.config.config.get("frog", {})
            return FrogProvider(
                api_key=frog_config.get("api_key") or os.environ.get("FROG_API_KEY"),
                username=frog_config.get("username") or os.environ.get("FROG_USERNAME"),
                sender_id=frog_config.get("sender_id", "CCMEWS"),
                use_test_api=frog_config.get("use_test_api", True)
            )
        elif provider_name == "africastalking":
            at_config = self.config.config.get("africastalking", {})
            return AfricasTalkingProvider(
                username=at_config.get("username") or os.environ.get("AT_USERNAME"),
                api_key=at_config.get("api_key") or os.environ.get("AT_API_KEY"),
                sender_id=at_config.get("sender_id", "CCMEWS")
            )
        elif provider_name == "twilio":
            tw_config = self.config.config.get("twilio", {})
            return TwilioProvider(
                account_sid=tw_config.get("account_sid") or os.environ.get("TWILIO_SID"),
                auth_token=tw_config.get("auth_token") or os.environ.get("TWILIO_TOKEN"),
                from_number=tw_config.get("from_number") or os.environ.get("TWILIO_FROM")
            )
        else:
            logger.warning(f"Unknown provider: {provider_name}, using test mode")
            return TestProvider()
    
    def format_alert(self, template_name: str, **kwargs) -> str:
        """Format an alert message using a template"""
        template = self.TEMPLATES.get(template_name, self.TEMPLATES["combined_alert"])
        kwargs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        return template.format(**kwargs)
    
    def should_alert(self, hazard_type: str, value: float) -> bool:
        """Check if value exceeds alert threshold"""
        thresholds = self.config.config.get("alert_thresholds", {})
        threshold = thresholds.get(f"{hazard_type}_risk", 0.5)
        return value >= threshold
    
    def send_alert(self, alert: AlertMessage, recipients: List[AlertRecipient] = None) -> Dict:
        """
        Send alert to all relevant recipients
        
        Args:
            alert: AlertMessage to send
            recipients: Optional list of recipients (uses config if not provided)
            
        Returns:
            Dictionary with send results
        """
        if not self.config.config.get("enabled", False):
            logger.warning("SMS alerts are disabled. Set 'enabled': true in config.")
            return {"sent": 0, "failed": 0, "disabled": True}
        
        if recipients is None:
            recipients = self.config.get_recipients(district=alert.location)
        
        if not recipients:
            logger.warning(f"No recipients configured for {alert.location}")
            return {"sent": 0, "failed": 0, "no_recipients": True}
        
        results = {"sent": 0, "failed": 0, "details": []}
        
        for recipient in recipients:
            result = self.provider.send(recipient.phone, alert.message)
            
            status = "sent" if result["success"] else "failed"
            self.logger.log_alert(
                alert, recipient, status,
                provider=self.config.config.get("provider"),
                message_id=result.get("message_id"),
                error=result.get("error")
            )
            
            if result["success"]:
                results["sent"] += 1
                logger.info(f"✓ Alert sent to {recipient.name} ({recipient.phone})")
            else:
                results["failed"] += 1
                logger.error(f"✗ Failed to send to {recipient.name}: {result.get('error')}")
            
            results["details"].append({
                "recipient": recipient.name,
                "phone": recipient.phone,
                "success": result["success"],
                "error": result.get("error")
            })
        
        return results
    
    def send_flood_alert(self, location: str, risk: float, rainfall_mm: float,
                        when: str = "Next 24 hours") -> Dict:
        """Send flood warning alert"""
        risk_level = "CRITICAL" if risk >= 0.65 else "HIGH" if risk >= 0.45 else "MODERATE"
        
        message = self.format_alert(
            "flood_warning",
            location=location,
            risk_level=risk_level,
            risk_pct=int(risk * 100),
            rainfall_mm=round(rainfall_mm, 1),
            when=when
        )
        
        alert = AlertMessage(
            alert_type="flood",
            severity="warning" if risk >= 0.45 else "watch",
            location=location,
            message=message,
            timestamp=datetime.now().isoformat(),
            data={"risk": risk, "rainfall_mm": rainfall_mm}
        )
        
        return self.send_alert(alert)
    
    def send_heat_alert(self, location: str, max_temp: float, heat_index: float,
                       duration: str = "1-2 days") -> Dict:
        """Send heat warning alert"""
        if max_temp >= 40:
            risk_level = "EXTREME"
        elif max_temp >= 37:
            risk_level = "HIGH"
        else:
            risk_level = "MODERATE"
        
        message = self.format_alert(
            "heat_warning",
            location=location,
            risk_level=risk_level,
            max_temp=round(max_temp, 1),
            heat_index=round(heat_index, 1),
            duration=duration
        )
        
        alert = AlertMessage(
            alert_type="heat",
            severity="warning" if max_temp >= 37 else "advisory",
            location=location,
            message=message,
            timestamp=datetime.now().isoformat(),
            data={"max_temp": max_temp, "heat_index": heat_index}
        )
        
        return self.send_alert(alert)
    
    def send_rainfall_forecast(self, location: str, rainfall_mm: float,
                              intensity: str, start_time: str,
                              duration_hours: int, probability: float) -> Dict:
        """Send rainfall forecast notification"""
        message = self.format_alert(
            "rainfall_forecast",
            location=location,
            intensity=intensity.upper(),
            rainfall_mm=round(rainfall_mm, 1),
            start_time=start_time,
            duration=duration_hours,
            probability=int(probability * 100)
        )
        
        alert = AlertMessage(
            alert_type="rainfall",
            severity="advisory",
            location=location,
            message=message,
            timestamp=datetime.now().isoformat(),
            data={"rainfall_mm": rainfall_mm, "intensity": intensity}
        )
        
        return self.send_alert(alert)


def setup_sample_config():
    """Create sample configuration with example recipients"""
    config = SMSConfig()
    
    # Add sample recipients (update with real numbers)
    sample_recipients = [
        {
            "name": "District Emergency Coordinator",
            "phone": "+233XXXXXXXXX",  # Update with real number
            "role": "Emergency Coordinator",
            "district": "North Tongu",
            "active": True
        },
        {
            "name": "NADMO District Officer",
            "phone": "+233XXXXXXXXX",  # Update with real number
            "role": "NADMO Officer",
            "district": "North Tongu",
            "active": True
        },
        {
            "name": "District Health Director",
            "phone": "+233XXXXXXXXX",  # Update with real number
            "role": "Health Director",
            "district": "North Tongu",
            "active": True
        }
    ]
    
    config.config["recipients"] = sample_recipients
    config.config["enabled"] = False  # Keep disabled until configured
    config.config["test_mode"] = True  # Start in test mode
    config.config["provider"] = "frog"  # Use Frog by Wigal
    config.save_config()
    
    print(f"""
✅ Sample configuration created at: {CONFIG_PATH}

═══════════════════════════════════════════════════════════════
FROG BY WIGAL SETUP (Recommended for Ghana)
═══════════════════════════════════════════════════════════════

1. Sign up at https://sms.wigal.com.gh
2. Get your API Key and Username from the dashboard
3. Register your Sender ID (e.g., "CCMEWS")
4. Update {CONFIG_PATH} with:
   
   "frog": {{
       "api_key": "your_api_key_here",
       "username": "your_username_here", 
       "sender_id": "CCMEWS",
       "use_test_api": false  // Set to false for production
   }}

5. Update recipient phone numbers (use +233XXXXXXXXX format)
6. Set "enabled": true and "test_mode": false

───────────────────────────────────────────────────────────────
ENVIRONMENT VARIABLES (Alternative)
───────────────────────────────────────────────────────────────
You can also set credentials via environment variables:
  export FROG_API_KEY="your_api_key"
  export FROG_USERNAME="your_username"

───────────────────────────────────────────────────────────────
PRICING (as of 2024)
───────────────────────────────────────────────────────────────
  GH₵ 5   → ~175 SMS
  GH₵ 10  → ~350 SMS  
  GH₵ 50  → ~1,759 SMS
  GH₵ 100 → ~3,533 SMS

───────────────────────────────────────────────────────────────
ALTERNATIVE PROVIDERS
───────────────────────────────────────────────────────────────
Africa's Talking: Set "provider": "africastalking"
Twilio: Set "provider": "twilio"
""")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CCMEWS SMS Alert System")
    parser.add_argument("--setup", action="store_true", help="Create sample configuration")
    parser.add_argument("--test", action="store_true", help="Send test alert")
    parser.add_argument("--history", action="store_true", help="Show recent alerts")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_sample_config()
    elif args.test:
        alert_system = AlertSystem()
        
        # Test flood alert
        print("\n📤 Sending test flood alert...")
        result = alert_system.send_flood_alert(
            location="Battor, North Tongu",
            risk=0.65,
            rainfall_mm=55,
            when="Tomorrow afternoon"
        )
        print(f"Result: {result}")
        
        # Test heat alert
        print("\n📤 Sending test heat alert...")
        result = alert_system.send_heat_alert(
            location="Adidome, North Tongu",
            max_temp=39.5,
            heat_index=44.2,
            duration="2-3 days"
        )
        print(f"Result: {result}")
        
    elif args.history:
        logger = AlertLogger()
        alerts = logger.get_recent_alerts(hours=48)
        
        print(f"\n📋 Recent Alerts (last 48 hours): {len(alerts)}")
        for alert in alerts[:10]:
            print(f"  [{alert['timestamp']}] {alert['alert_type'].upper()} to {alert['recipient_name']} - {alert['status']}")
    else:
        parser.print_help()
