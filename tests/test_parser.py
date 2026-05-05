from llm.parse import extract_json_from_markdown

text = """```json
{
  "floor_plan": {
    "scale_anchor": "Standard interior door width (approx. 81cm / 32in) represented by ~75 units on the 1000x1000 grid.",
    "rooms": [
      {
        "name": "Kitchen/Dining Area",
        "bounding_box": [0, 0, 445, 450],
        "objects": [
          {
            "type": "Kitchen Counter with Sink",
            "bounding_box": [10, 0, 100, 90]
          },
          {
            "type": "Round Dining Table with Chairs",
            "bounding_box": [150, 150, 3"""

print("Testing robust parser:")
try:
    res = extract_json_from_markdown(text)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
