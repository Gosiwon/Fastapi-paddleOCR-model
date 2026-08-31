from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="korean",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

def recognize(image_path: str):
    return ocr.predict(image_path)

