from paddleocr import TextRecognition

recognizer = TextRecognition(
    model_name="korean_PP-OCRv5_mobile_rec",
    device="cpu"
)

results = recognizer.predict(
    input="app/test/samples/bbox/bbox_output.png",
    batch_size = 1
)

for result in results:
    result.print()