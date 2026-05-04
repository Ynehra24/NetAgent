import os
import sys
import json

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__line__ if '__line__' in globals() else __file__))))

from steps.parse import extract_floorplan_features

def test_extraction_pipeline():
    # If a path is provided via command line, use it
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        # Look for any image in the data/images or data directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        test_image_path = None
        for img_dir in [os.path.join(base_dir, "data", "images"), os.path.join(base_dir, "data")]:
            if os.path.exists(img_dir):
                for file in os.listdir(img_dir):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        test_image_path = os.path.join(img_dir, file)
                        break
            if test_image_path:
                break

    if not test_image_path or not os.path.exists(test_image_path):
        print(f"No test image found. Please pass an image path as an argument.")
        return

    print(f"Testing extraction with image: {test_image_path}")
    
    try:
        # Run the extraction
        result_json = extract_floorplan_features(test_image_path)
        
        # Print the structured output
        print("\n--- Extraction Result ---")
        print(json.dumps(result_json, indent=2))
        print("-------------------------\n")
        print("Success! JSON was parsed correctly.")

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_extraction_pipeline()
