NLP_TASKS = {
    "text_binary_classification": 1,
    # "text_multi_class_classification": 2,
    # "text_entity_extraction": 4,
    # "text_extractive_question_answering": 5,
    # "text_summarization": 8,
    # "text_single_column_regression": 10,
    # "speech_recognition": 11,
    # "natural_language_inference": 22,
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
    # **VISION_TASKS,
    # **TABULAR_TASKS,
}

COLUMN_MAPPING = {
    "text_binary_classification": ("text", "label"),
    "text_multi_class_classification": ("text", "label"),
    "text_entity_extraction": ("text", "tokens"),
    "text_extractive_question_answering": ("text", "context", "question", "answer"),
    "text_summarization": ("text", "summary"),
    "text_single_column_regression": ("text", "label"),
    "speech_recognition": ("audio", "text"),
    "natural_language_inference": ("premise", "hypothesis", "label"),
    "image_binary_classification": ("image", "label"),
    "image_multi_class_classification": ("image", "label"),
    "image_single_column_regression": ("image", "label"),
    # "dreambooth": ("image", "label"),
    "tabular_binary_classification": ("id", "label"),
    "tabular_multi_class_classification": ("id", "label"),
    "tabular_multi_label_classification": ("id", "label"),
    "tabular_single_column_regression": ("id", "label"),
}
