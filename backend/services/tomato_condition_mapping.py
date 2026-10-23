
# backend/services/tomato_condition_mapping.py

TOMATO_CLASS_MAPPING = {
    "Bacterial_spot": {
        "condition": "Bacterial spot",
        "category": "disease",
    },
    "Early_blight": {
        "condition": "Early blight",
        "category": "disease",
    },
    "Healthy": {
        "condition": "Healthy tomato leaf",
        "category": "healthy",
    },
    "Jassids_and_mites": {
        "condition": "Jassids and mites",
        "category": "pest",
    },
    "Late_blight": {
        "condition": "Late blight",
        "category": "disease",
    },
    "Leaf_Mold": {
        "condition": "Leaf mold",
        "category": "disease",
    },
    "Leaf_miner": {
        "condition": "Leaf miner",
        "category": "pest",
    },
    "Mites": {
        "condition": "Mite infestation",
        "category": "pest",
    },
    "Nitrogen_deficiency": {
        "condition": "Nitrogen deficiency",
        "category": "nutrient_deficiency",
    },
    "Pottassium_deficiency": {
        "condition": "Potassium deficiency",
        "category": "nutrient_deficiency",
    },
    "Magnesium_deficiency": {
        "condition": "Magnesium deficiency",
        "category": "nutrient_deficiency",
    },
    "Septoria_leaf_spot": {
        "condition": "Septoria leaf spot",
        "category": "disease",
    },
    "Spider_mites": {
        "condition": "Spider mite infestation",
        "category": "pest",
    },
    "Target_Spot": {
        "condition": "Target spot",
        "category": "disease",
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "condition": "Tomato yellow leaf curl virus",
        "category": "disease",
    },
    "Tomato_mosaic_virus": {
        "condition": "Tomato mosaic virus",
        "category": "disease",
    },
}


def map_tomato_prediction(class_name: str) -> dict:
    """Convert a model class into a canonical condition."""

    mapped = TOMATO_CLASS_MAPPING.get(class_name)

    if mapped is None:
        return {
            "original_class": class_name,
            "condition": None,
            "category": None,
            "supported": False,
        }

    return {
        "original_class": class_name,
        **mapped,
        "supported": True,
    }
