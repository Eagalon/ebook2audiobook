#!/usr/bin/env python3
"""
Test script for piper-tts integration in ebook2audiobook
This script tests the basic functionality without requiring the full app environment.
"""

import sys
import os
import tempfile
import wave

# Add the lib directory to Python path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

def test_piper_basic():
    """Test basic piper-tts functionality"""
    print("Testing basic piper-tts functionality...")
    
    try:
        from piper import PiperVoice
        print("✓ piper-tts package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import piper-tts: {e}")
        return False
    
    try:
        # Test downloading and loading a voice model
        print("Testing voice model download and loading...")
        
        # Download a small model for testing
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'piper.download_voices', 'en_US-lessac-medium'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Voice model downloaded successfully")
        else:
            print(f"✗ Voice model download failed: {result.stderr}")
            print("Continuing with test assuming model is already available...")
        
        # Try to find the downloaded model
        home_dir = os.path.expanduser("~")
        model_paths = [
            os.path.join(home_dir, ".local/share/piper-voices/en_US-lessac-medium"),
            "/tmp/piper-voices/en_US-lessac-medium"
        ]
        
        model_file = None
        config_file = None
        
        for path in model_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.endswith('.onnx'):
                        model_file = os.path.join(path, file)
                    elif file.endswith('.onnx.json'):
                        config_file = os.path.join(path, file)
                if model_file and config_file:
                    break
        
        if not model_file or not config_file:
            print(f"✗ Model files not found in expected locations: {model_paths}")
            return False
        
        print(f"✓ Found model files: {model_file}, {config_file}")
        
        # Load the voice
        voice = PiperVoice.load(model_file, config_path=config_file, use_cuda=False)
        print("✓ Voice loaded successfully")
        
        # Test synthesis
        test_text = "Hello, this is a test of piper text to speech synthesis."
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with wave.open(temp_path, 'wb') as wav_file:
                voice.synthesize(test_text, wav_file)
            
            # Check if file was created and has content
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                print(f"✓ Audio synthesis successful: {temp_path}")
                print(f"  File size: {os.path.getsize(temp_path)} bytes")
                return True
            else:
                print("✗ Audio file was not created or is empty")
                return False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tts_engines_integration():
    """Test integration with ebook2audiobook TTS engine system"""
    print("\nTesting integration with ebook2audiobook TTS engine system...")
    
    try:
        # Test import of models
        from models import TTS_ENGINES, default_engine_settings
        print("✓ TTS models imported successfully")
        
        # Check PIPER engine is available
        if 'PIPER' in TTS_ENGINES:
            print(f"✓ PIPER engine available: {TTS_ENGINES['PIPER']}")
        else:
            print("✗ PIPER engine not found in TTS_ENGINES")
            return False
        
        # Check PIPER configuration
        if TTS_ENGINES['PIPER'] in default_engine_settings:
            config = default_engine_settings[TTS_ENGINES['PIPER']]
            print(f"✓ PIPER configuration found: {config}")
        else:
            print("✗ PIPER configuration not found in default_engine_settings")
            return False
        
        # Test TTSManager import
        from classes.tts_manager import TTSManager
        print("✓ TTSManager imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Piper-TTS Integration Test")
    print("=" * 40)
    
    # Test basic piper functionality
    piper_test = test_piper_basic()
    
    # Test integration with ebook2audiobook
    integration_test = test_tts_engines_integration()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Piper-TTS Basic: {'✓ PASS' if piper_test else '✗ FAIL'}")
    print(f"Integration: {'✓ PASS' if integration_test else '✗ FAIL'}")
    
    if piper_test and integration_test:
        print("\n🎉 All tests passed! Piper-TTS integration is working.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())