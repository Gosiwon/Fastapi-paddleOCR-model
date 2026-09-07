import argparse
import json
import shutil
from pathlib import Path

def create_image_index(image_root: Path):
    return {
        image_path.stem: image_path
        for image_path in image_root.rglob("*")
    }

def convert_bbox(bbox):
    x_values=bbox["x"]
    y_values=bbox["y"]

    x1 = min(x_values)
    y1 = min(y_values)
    x2 = max(x_values)
    y2 = max(y_values)

    return {
        "transcription": bbox["data"],
        "points": [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]
    }

def create_split(
    image_root: Path,
    label_root: Path,
    output_root:Path,
    split: str,
    limit: int,
    offset: int
):
    image_paths = [
        path
        for path in image_root.rglob("*")
    ]

    label_index = {
        path.stem: path
        for path in label_root.rglob("*.json")
    }

    output_image_dir = output_root / "images" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)

    output_lines = []
    unmatched_images = []

    for image_path in sorted(image_paths):
        label_path = label_index.get(image_path.stem)

        if label_path is None:
            unmatched_images.append(image_path.name)
            continue

        with label_path.open("r", encoding="utf-8") as file:
            label = json.load(file)

        raw_boxes = label.get("bbox")

        if not raw_boxes:
            continue

        converted_boxes = [
            convert_bbox(box)
            for box in raw_boxes
            if box.get("data") and box.get("x") and box.get("y")
        ]

        if not converted_boxes:
            continue

        destination = output_image_dir / image_path.name
        shutil.copy2(image_path, destination)

        relative_path = destination.relative_to(output_root).as_posix()

        output_lines.append(
            relative_path
            + "\t"
            + json.dumps(converted_boxes, ensure_ascii=False)
        )

        if len(output_lines) >= limit:
            break


    output_file = output_root / f"{split}.txt"
    output_file.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8"
    )

    print(f"{split} 생성 개수: {len(output_lines)}")
    print(f"라벨이 없는 이미지: {len(unmatched_images)}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    create_split(
        args.train_images,
        args.train_labels,
        args.output,
        split="train",
        limit=20,
        offset=0
    )

    create_split(
        args.train_images,
        args.train_labels,
        args.output,
        split="val",
        limit=5,
        offset=20
    )


if __name__ == "__main__":
    main()

