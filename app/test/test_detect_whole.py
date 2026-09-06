from pathlib import Path

from app.ml.paddle_engine import recognize

image_path = (
    Path(__file__).resolve().parent
    / "samples"
    / "whole"
    / "IMG_OCR_53_4PO_09451_small.png"
)

results = recognize(str(image_path))

for result in results:
    result.save_to_img("app/test/whole_output/detection")
    result.save_to_json("app/test/whole_output/detection")