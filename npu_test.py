import cv2
import numpy as np
import time
from PIL import Image

def test_npu():
    print("Testing Orange Pi NPU...")
    
    # Test 1: Basic NPU detection
    try:
        import cv2
        # Check OpenCV build information to see if NPU is enabled
        build_info = cv2.getBuildInformation()
        print("OpenCV Build Info:")
        print(build_info)
        
        # Look for NPU-related information in build info
        if "NPU" in build_info or "DPU" in build_info:
            print("✓ NPU support detected in OpenCV")
        else:
            print("✗ NPU support not found in OpenCV build")
    except Exception as e:
        print(f"Error checking NPU support: {e}")
    
    # Test 2: Try to use NPU backend in OpenCV
    try:
        # Create a simple neural network
        net = cv2.dnn.readNetFromONNX("sample_model.onnx")  # This will fail if file doesn't exist
        
        # Try setting backend to NPU/DNN
        backends = [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_CUDA]
        targets = [cv2.dnn.DNN_TARGET_CPU]
        
        # For Orange Pi NPU, you might have these targets available
        try:
            targets.append(cv2.dnn.DNN_TARGET_NPU)
        except AttributeError:
            print("NPU target not available in this OpenCV build")
        
        for backend in backends:
            for target in targets:
                try:
                    net.setPreferableBackend(backend)
                    net.setPreferableTarget(target)
                    print(f"✓ Backend {backend}, Target {target} - Configured successfully")
                except Exception as e:
                    print(f"✗ Backend {target}, Target {target} - Failed: {e}")
    except Exception as e:
        print(f"Could not test network backend configuration: {e}")
        print("This might be because a model file is required.")
    
    # Test 3: Simple image processing to verify hardware acceleration
    try:
        # Create a sample image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Measure processing time for a computationally intensive operation
        start_time = time.time()
        for i in range(10):
            # Apply Gaussian blur - computationally intensive
            blurred = cv2.GaussianBlur(img, (15, 15), 0)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"Average processing time for Gaussian blur: {avg_time:.4f} seconds")
        
        if avg_time < 0.05:  # Threshold may need adjustment
            print("✓ Hardware acceleration likely working (fast processing)")
        else:
            print("✗ Processing time suggests no hardware acceleration")
    except Exception as e:
        print(f"Error in image processing test: {e}")
    
    # Test 4: Check for specific NPU libraries
    npu_libs = [
        "librknn",
        "rknn",
        "libnpu", 
        "Ascend"
    ]
    
    found_libs = []
    for lib in npu_libs:
        try:
            # Try different import patterns
            if lib == "rknn":
                import rknn
                found_libs.append(lib)
            elif lib == "librknn":
                import rknn.api as rknn
                found_libs.append("rknn")
        except ImportError:
            pass
        except Exception as e:
            found_libs.append(f"{lib} (error: {e})")
    
    if found_libs:
        print(f"✓ Found NPU-related libraries: {', '.join(found_libs)}")
    else:
        print("✗ No NPU-specific libraries detected")
    
    print("\nNPU testing completed.")

if __name__ == "__main__":
    test_npu()