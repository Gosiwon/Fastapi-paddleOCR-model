import cv2

from app.ml.paddle_engine import recognize

image_path = "app/test/samples/IMG_OCR_53_4PO_09451.png"
output_path = "app/test/samples/bbox/bbox_output.png"

image = cv2.imread(image_path)
print("현재 이미지 크기:", image.shape)

x_values = [717, 717, 897, 897]
y_values = [832, 932, 832, 932]

padding = 30

x1 = max(0, min(x_values) - padding)
y1 = max(0, min(y_values) - padding)
x2 = min(image.shape[1], max(x_values) + padding)
y2 = min(image.shape[0], max(y_values) + padding)

crop = image[y1:y2, x1:x2]
cv2.imwrite(output_path, crop)

results = recognize(output_path)

for result in results:
    result.save_to_img("app/test/bbox_output")
    result.save_to_json("app/test/bbox_output")

