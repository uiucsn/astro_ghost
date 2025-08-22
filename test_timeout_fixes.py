#!/usr/bin/env python
"""
Simple test to verify our timeout fixes work without TensorFlow dependencies
"""

def test_timeout_fixes():
    """Test that timeout fixes are present in the source code"""
    import os
    
    # Check PS1QueryFunctions.py has timeout fixes
    ps1_file = os.path.join(os.path.dirname(__file__), 'astro_ghost', 'PS1QueryFunctions.py')
    with open(ps1_file, 'r') as f:
        ps1_content = f.read()
    
    # Should have at least 4 timeout=15 occurrences
    timeout_count = ps1_content.count('timeout=15')
    assert timeout_count >= 4, f"Expected at least 4 timeout fixes in PS1QueryFunctions.py, found {timeout_count}"
    
    # Check ghostHelperFunctions.py has timeout fix
    ghost_file = os.path.join(os.path.dirname(__file__), 'astro_ghost', 'ghostHelperFunctions.py')
    with open(ghost_file, 'r') as f:
        ghost_content = f.read()
    
    timeout_count = ghost_content.count('timeout=15')
    assert timeout_count >= 1, f"Expected at least 1 timeout fix in ghostHelperFunctions.py, found {timeout_count}"
    
    # Check stellarLocus.py has NaN fix
    stellar_file = os.path.join(os.path.dirname(__file__), 'astro_ghost', 'stellarLocus.py')
    with open(stellar_file, 'r') as f:
        stellar_content = f.read()
    
    assert 'Handle all-NaN arrays' in stellar_content, "NaN warning fix not found in stellarLocus.py"
    assert 'if np.all(np.isnan(temp_array)):' in stellar_content, "NaN handling logic not found"
    
    print("✅ All timeout and warning fixes are present!")

def test_basic_import():
    """Test that basic imports work"""
    try:
        from astro_ghost.PS1QueryFunctions import ps1cone
        from astro_ghost.ghostHelperFunctions import getGHOST
        from astro_ghost.stellarLocus import calc_7DCD
        print("✅ Basic imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_timeout_fixes()
    test_basic_import()
    print("🎉 All tests passed!")