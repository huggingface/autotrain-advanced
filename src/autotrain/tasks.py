NLP_TASKS = {
    "text_binary_classification": 1,
    "text_multi_class_classification": 2,
    "text_token_classification": 4,
    "text_extractive_question_answering": 5,
    "text_summarization": 8,
    "text_single_column_regression": 10,
    "speech_recognition": 11,
    "natural_language_inference": 22,
    "lm_training": 9,
    "seq2seq": 28,  # 27 is reserved for generic training
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
    "tabular": 26,
}


TASKS = {
    **NLP_TASKS,
    **VISION_TASKS,
    **TABULAR_TASKS,
}

COLUMN_MAPPING = {
    "text_binary_classification": ("text", "label"),
    "text_multi_class_classification": ("text", "label"),
    "text_token_classification": ("tokens", "tags"),
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
    "lm_training": ("text", "prompt_start", "prompt", "context", "response"),
}

TASK_TYPE_MAPPING = {
    "text_binary_classification": "Natural Language Processing",
    "text_multi_class_classification": "Natural Language Processing",
    "text_token_classification": "Natural Language Processing",
    "text_extractive_question_answering": "Natural Language Processing",
    "text_summarization": "Natural Language Processing",
    "text_single_column_regression": "Natural Language Processing",
    "lm_training": "Natural Language Processing",
    "speech_recognition": "Natural Language Processing",
    "natural_language_inference": "Natural Language Processing",
    "image_binary_classification": "Computer Vision",
    "image_multi_class_classification": "Computer Vision",
    "image_single_column_regression": "Computer Vision",
    "dreambooth": "Computer Vision",
    "tabular_binary_classification": "Tabular",
    "tabular_multi_class_classification": "Tabular",
    "tabular_multi_label_classification": "Tabular",
    "tabular_single_column_regression": "Tabular",
}
