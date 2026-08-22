"""
COCO 80-Class to Category Mapper Module (Phase 6.1)

Categorizes COCO object class names into 4 primary categories:
- PERSON
- ANIMAL
- OBJECT
- VEHICLE
"""

COCO_CATEGORIES = {
    "PERSON": {
        "person"
    },
    "ANIMAL": {
        "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
    },
    "OBJECT": {
        "backpack", "umbrella", "handbag", "tie", "suitcase", "bottle", "cup", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    },
    "VEHICLE": {
        "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
    }
}


def categorize_class(class_name: str) -> str:
    """
    Maps COCO class name to category string: 'PERSON', 'ANIMAL', 'OBJECT', or 'VEHICLE'.
    Defaults to 'OBJECT' for unrecognized classes.
    """
    name = str(class_name).lower().strip()
    if name in COCO_CATEGORIES["PERSON"]:
        return "PERSON"
    elif name in COCO_CATEGORIES["ANIMAL"]:
        return "ANIMAL"
    elif name in COCO_CATEGORIES["VEHICLE"]:
        return "VEHICLE"
    else:
        return "OBJECT"
