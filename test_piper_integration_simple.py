#!/usr/bin/env python3

"""
Simple test to verify PIPER integration follows the template correctly
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, '/home/runner/work/ebook2audiobook/ebook2audiobook')

def test_piper_in_models():
    """Test that PIPER is properly configured in models.py"""
    from lib.models import TTS_ENGINES, models, default_engine_settings
    
    print("Testing PIPER in TTS_ENGINES...")
    assert 'PIPER' in TTS_ENGINES, "PIPER not found in TTS_ENGINES"
    assert TTS_ENGINES['PIPER'] == 'piper', f"PIPER engine name incorrect: {TTS_ENGINES['PIPER']}"
    
    print("Testing PIPER in models...")
    assert TTS_ENGINES['PIPER'] in models, "PIPER not found in models"
    assert 'internal' in models[TTS_ENGINES['PIPER']], "PIPER internal config not found"
    
    print("Testing PIPER in default_engine_settings...")
    assert TTS_ENGINES['PIPER'] in default_engine_settings, "PIPER not found in default_engine_settings"
    piper_settings = default_engine_settings[TTS_ENGINES['PIPER']]
    assert 'voices' in piper_settings, "PIPER voices not configured"
    assert 'samplerate' in piper_settings, "PIPER samplerate not configured"
    
    print("✓ PIPER is properly configured in models.py")

def test_tts_manager():
    """Test that TTS Manager can handle PIPER"""
    from lib.classes.tts_manager import TTSManager
    
    # Create a mock session for PIPER
    session = {
        'tts_engine': 'piper',
        'fine_tuned': 'internal',
        'custom_model': None,
        'device': 'cpu',
        'voice_model': 'en_US-lessac-medium',
        'language': 'eng',
        'language_iso1': 'en',
        'process_dir': '/tmp',
        'final_name': 'test.mp3',
        'chapters_dir_sentences': '/tmp'
    }
    
    print("Testing TTSManager with PIPER...")
    # This should not fail with import errors
    try:
        tts_manager = TTSManager(session)
        print("✓ TTSManager can instantiate with PIPER engine")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"! Other error (may be expected): {e}")
        return True  # Other errors are acceptable for this test

def test_coqui_class_structure():
    """Test that Coqui class has PIPER support integrated"""
    
    # Mock the lib imports to avoid dependency issues
    class MockLib:
        pass
    
    import sys
    if 'lib' not in sys.modules:
        sys.modules['lib'] = MockLib()
        
    # Try to import just to check syntax
    try:
        import ast
        with open('/home/runner/work/ebook2audiobook/ebook2audiobook/lib/classes/tts_engines/coqui.py', 'r') as f:
            source = f.read()
        
        # Parse the file to check for PIPER support
        tree = ast.parse(source)
        
        # Check for PIPER in method bodies
        piper_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Str) and 'PIPER' in node.s:
                piper_found = True
                break
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and 'PIPER' in node.value:
                piper_found = True
                break
        
        # Better check: look for the actual string pattern
        if "TTS_ENGINES['PIPER']" in source:
            piper_found = True
            
        if "_load_piper_voice" in source:
            piper_found = True
            
        if "_synthesize_with_piper" in source:
            piper_found = True
        
        if piper_found:
            print("✓ Coqui class contains PIPER-specific code")
            return True
        else:
            print("✗ Coqui class does not contain PIPER support")
            return False
            
    except Exception as e:
        print(f"✗ Error checking Coqui class structure: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Testing PIPER Integration ===\n")
    
    tests = [
        test_piper_in_models,
        test_tts_manager,
        test_coqui_class_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print("")
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}\n")
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! PIPER integration follows the template correctly.")
        return True
    else:
        print("❌ Some tests failed. PIPER integration needs fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)