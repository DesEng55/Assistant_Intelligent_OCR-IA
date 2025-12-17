"""
Diagnostic script to debug API configuration issues
Run this to check what's wrong with your setup
"""

import os
import sys
from pathlib import Path

print("="*60)
print("🔍 DIAGNOSTIC SCRIPT - OCR Assistant")
print("="*60)

# 1. Check current directory
print("\n1. Current Working Directory:")
print(f"   {os.getcwd()}")

# 2. Check if .env file exists
print("\n2. Checking for .env file:")
env_path = Path(".env")
if env_path.exists():
    print(f"   ✅ .env file found at: {env_path.absolute()}")
    print(f"   File size: {env_path.stat().st_size} bytes")
    
    # Read and display (masked)
    with open(env_path, 'r') as f:
        content = f.read()
        print(f"\n   .env file content (first 100 chars):")
        print(f"   {content[:100]}")
        
        if "HUGGINGFACE_API_KEY" in content:
            print("   ✅ HUGGINGFACE_API_KEY found in .env")
        else:
            print("   ❌ HUGGINGFACE_API_KEY NOT found in .env")
else:
    print(f"   ❌ .env file NOT found")
    print(f"   Expected location: {env_path.absolute()}")

# 3. Try loading with python-dotenv
print("\n3. Testing python-dotenv:")
try:
    from dotenv import load_dotenv
    print("   ✅ python-dotenv is installed")
    
    # Load .env
    load_dotenv()
    print("   ✅ load_dotenv() executed")
    
except ImportError:
    print("   ❌ python-dotenv NOT installed")
    print("   Run: pip install python-dotenv")
    sys.exit(1)

# 4. Check environment variables
print("\n4. Checking Environment Variables:")
hf_key = os.getenv("HUGGINGFACE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if hf_key:
    print(f"   ✅ HUGGINGFACE_API_KEY loaded: {hf_key[:10]}...{hf_key[-5:]}")
    print(f"      Length: {len(hf_key)} characters")
    print(f"      Starts with 'hf_': {hf_key.startswith('hf_')}")
else:
    print("   ❌ HUGGINGFACE_API_KEY is None or empty")

if openai_key:
    print(f"   ✅ OPENAI_API_KEY loaded: {openai_key[:10]}...{openai_key[-5:]}")
else:
    print("   ℹ️  OPENAI_API_KEY not set (optional)")

# 5. Test Hugging Face API
if hf_key:
    print("\n5. Testing Hugging Face API Connection:")
    try:
        import requests
        
        api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
        headers = {"Authorization": f"Bearer {hf_key}"}
        
        test_payload = {
            "inputs": "Hello, how are you?",
            "parameters": {"max_new_tokens": 10}
        }
        
        print(f"   Sending test request to: {api_url}")
        response = requests.post(api_url, headers=headers, json=test_payload, timeout=10)
        
        print(f"   Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ API KEY IS VALID AND WORKING!")
            print(f"   Response: {response.json()}")
        elif response.status_code == 401:
            print("   ❌ INVALID API KEY - Authentication failed")
            print("   The key may have been revoked or is incorrect")
        elif response.status_code == 403:
            print("   ❌ ACCESS DENIED - Key doesn't have permission")
        else:
            print(f"   ⚠️  Unexpected response: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("   ⚠️  Request timeout - API might be loading the model")
        print("   This is normal for first request, try again")
    except Exception as e:
        print(f"   ❌ Error testing API: {str(e)}")
else:
    print("\n5. ⏭️  Skipping API test (no key found)")

# 6. Check Config class
print("\n6. Testing Config Class:")
try:
    from config import Config
    config = Config()
    
    print(f"   ✅ Config imported successfully")
    
    if hasattr(config, 'HUGGINGFACE_API_KEY'):
        print(f"   ✅ HUGGINGFACE_API_KEY attribute exists")
        if config.HUGGINGFACE_API_KEY:
            print(f"      Value: {config.HUGGINGFACE_API_KEY[:10]}...")
        else:
            print(f"      ⚠️  Value is empty string")
    else:
        print(f"   ❌ HUGGINGFACE_API_KEY attribute MISSING")
        
    if hasattr(config, 'OPENAI_API_KEY'):
        print(f"   ✅ OPENAI_API_KEY attribute exists")
    else:
        print(f"   ❌ OPENAI_API_KEY attribute MISSING")
        
except ImportError as e:
    print(f"   ❌ Cannot import Config: {e}")
except Exception as e:
    print(f"   ❌ Error with Config: {e}")

# 7. Check QwenIntegration
print("\n7. Testing QwenIntegration:")
try:
    from qwen_integration import QwenIntegration
    qwen = QwenIntegration()
    
    print(f"   ✅ QwenIntegration imported")
    print(f"   Mode: {qwen.mode}")
    print(f"   Has hf_api_key: {qwen.hf_api_key is not None}")
    
    if qwen.mode == "huggingface_api":
        print("   ✅ Configured to use Hugging Face API")
    elif qwen.mode == "openai_api":
        print("   ℹ️  Configured to use OpenAI API")
    elif qwen.mode == "local":
        print("   ℹ️  Configured to use local model")
    else:
        print("   ❌ No valid mode configured")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# 8. Summary
print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)

issues = []
if not env_path.exists():
    issues.append("❌ .env file is missing")
if not hf_key:
    issues.append("❌ HUGGINGFACE_API_KEY not loaded from .env")
if hf_key and not hf_key.startswith('hf_'):
    issues.append("⚠️  API key doesn't start with 'hf_'")

if issues:
    print("\n🔴 ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")
    
    print("\n💡 SOLUTIONS:")
    print("   1. Create .env file in project root if missing")
    print("   2. Add: HUGGINGFACE_API_KEY=hf_your_key_here")
    print("   3. Get new key from: https://huggingface.co/settings/tokens")
    print("   4. Restart the application")
else:
    print("\n✅ Everything looks good!")
    print("   If the app still doesn't work, check the API test results above.")

print("\n" + "="*60)