NLP_TASKS = {
    "binary_classification": 1,
    "multi_class_classification": 2,
    "entity_extraction": 4,
    "extractive_question_answering": 5,
    "summarization": 8,
    "single_column_regression": 10,
    "speech_recognition": 11,
    "natural_language_inference": 22,
}

VISION_TASKS = {
    "image_binary_classification": 17,
    "image_multi_class_classification": 18,
    "image_single_column_regression": 24,
    "dreambooth": 25,
}

TABULAR_TASKS = {
    "tabular_binary_classification": 13,
    "tabular_multi_class_classification": 14,
    "tabular_multi_label_classification": 15,
    "tabular_single_column_regression": 16,
}


TASKS = {
    **NLP_TASKS,
    **VISION_TASKS,
    **TABULAR_TASKS,
}
