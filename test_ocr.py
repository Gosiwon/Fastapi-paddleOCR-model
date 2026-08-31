from app.ml.paddle_engine import recognize

results = recognize("app/samples/img.png")

for result in results:
    result.print()