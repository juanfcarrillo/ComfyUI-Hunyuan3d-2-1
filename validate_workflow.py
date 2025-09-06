#!/usr/bin/env python3
"""
Validation script for the optimized Hunyuan 3D workflow
Tests basic functionality without requiring actual model files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from manual_workflow import (
            ManualHunyuan3DWorkflow,
            ManualHunyuan3DTextureWorkflow,
            CompleteHunyuan3DWorkflow,
            EnhancedHunyuan3DWorkflow
        )
        print("✓ All workflow classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_workflow_instantiation():
    """Test that workflows can be instantiated"""
    try:
        from manual_workflow import EnhancedHunyuan3DWorkflow

        workflow = EnhancedHunyuan3DWorkflow()
        print("✓ Enhanced workflow instantiated successfully")

        # Check that all required components are present
        assert hasattr(workflow, 'mesh_workflow'), "Missing mesh_workflow"
        assert hasattr(workflow, 'texture_workflow'), "Missing texture_workflow"
        assert hasattr(workflow, 'mesh_decimator'), "Missing mesh_decimator"
        print("✓ All workflow components present")

        return True
    except Exception as e:
        print(f"✗ Instantiation error: {e}")
        return False

def test_parameter_validation():
    """Test that parameters are properly configured for 16GB VRAM"""
    try:
        from manual_workflow import EnhancedHunyuan3DWorkflow
        import inspect

        workflow = EnhancedHunyuan3DWorkflow()
        sig = inspect.signature(workflow.run_enhanced_workflow)

        # Check that memory-optimized defaults are set
        params = sig.parameters

        # Check mesh parameters
        mesh_params = params.get('mesh_params')
        assert mesh_params.default is None, "mesh_params should default to None"

        # Check texture parameters
        texture_params = params.get('texture_params')
        assert texture_params.default is None, "texture_params should default to None"

        # Check decimation parameters
        assert params['target_face_count'].default == 15000, "target_face_count should be 15000 for 16GB VRAM"
        assert params['enable_decimation'].default == True, "decimation should be enabled by default"

        print("✓ Parameters optimized for 16GB VRAM")
        return True
    except Exception as e:
        print(f"✗ Parameter validation error: {e}")
        return False

def main():
    """Run all validation tests"""
    print("Running Hunyuan 3D Workflow Validation Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_workflow_instantiation,
        test_parameter_validation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"FAILED: {test.__name__}")

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All validation tests passed!")
        print("\nThe optimized workflow is ready to use.")
        print("Run with: python manual_workflow.py --workflow enhanced --input-image your_image.png")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
