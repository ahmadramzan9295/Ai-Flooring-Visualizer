
import numpy as np
import cv2
import sys
import os

# Add local directory to path to import core
sys.path.append(os.getcwd())

import core

# Mock SamPredictor to avoid loading heavy model
class MockPredictor:
    def set_image(self, image):
        pass
    
    def predict(self, point_coords=None, point_labels=None, multimask_output=True):
        print(f"Predict called with points: {point_coords}")
        print(f"Predict called with labels: {point_labels}")
        
        # Verify labels are generated correctly
        if point_coords is not None:
            if point_labels is None:
                print("FAIL: point_labels should not be None when point_coords are active")
                sys.exit(1)
            if len(point_labels) != len(point_coords):
                 print(f"FAIL: len(labels) {len(point_labels)} != len(coords) {len(point_coords)}")
                 sys.exit(1)
            if not np.all(point_labels == 1):
                 print("FAIL: All labels should be 1 (foreground)")
                 sys.exit(1)
        
        # Return dummy masks (N, H, W), scores, logits
        h, w = 100, 100
        masks = np.zeros((3, h, w), dtype=bool)
        scores = np.array([0.9, 0.8, 0.7])
        logits = np.zeros((3, h, w))
        return masks, scores, logits

def test_multi_point_logic():
    print("Testing multi-point logic in core.py...")
    
    # 1. Setup dummy data
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    predictor = MockPredictor()
    
    # 2. Test with explicit points (simulating user clicks)
    points = np.array([[10, 10], [20, 20], [50, 50]])
    
    # 3. Call get_floor_mask
    try:
        core.get_floor_mask(image, predictor, point_coords=points)
        print("SUCCESS: get_floor_mask handled multiple points correctly.")
    except Exception as e:
        print(f"FAIL: get_floor_mask raised exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_multi_point_logic()
