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
    "sentence_transformers": 30,
}

VISION_TASKS = {
    "image_binary_classification": 17,
    "image_multi_class_classification": 18,
    "image_single_column_regression": 24,
    "image_object_detection": 29,
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
