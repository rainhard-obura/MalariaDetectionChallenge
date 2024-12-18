from typing import Dict, List

# Define class names and their corresponding indices
CATEGORIES: List[str] = ["Trophozoite", "WBC", "Background"]  # For the model
CATEGORIES_MAP: List[str] = ["WBC", "Trophozoite", "NEG"]  # For the mAP calculation

ID2LABEL: Dict[int, str] = {
    index: category for index, category in enumerate(CATEGORIES)
}
LABEL2ID: Dict[str, int] = {category: index for index, category in ID2LABEL.items()}

# Define colors for each class
CLASS_COLORS: Dict[str, str] = {
    "Trophozoite": "turquoise",
    "WBC": "yellow",
    "Background": "purple",
}

CLASSES: Dict[int, str] = {0: "Trophozoite", 1: "WBC", 2: "NEG", 3: "Background"}
