from functools import partial

from autotrain import logger


def _prepare_dataset(examples, tokenizer, config):
    # taken from:
    # https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
    # and modified for AutoTrain
    pad_on_right = tokenizer.padding_side == "right"
    tokenized_examples = tokenizer(
        examples[config.question_column if pad_on_right else config.text_column],
        examples[config.text_column if pad_on_right else config.question_column],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=config.max_seq_length,
        stride=config.max_doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        if tokenizer.cls_token_id in input_ids:
            cls_index = input_ids.index(tokenizer.cls_token_id)
        elif tokenizer.bos_token_id in input_ids:
            cls_index = input_ids.index(tokenizer.bos_token_id)
        else:
            cls_index = 0

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples[config.answer_column][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


class ExtractiveQuestionAnsweringDataset:
    """
    A dataset class for extractive question answering tasks.

    Args:
        data (Dataset): The dataset to be processed.
        tokenizer (PreTrainedTokenizer): The tokenizer to be used for processing the data.
        config (dict): Configuration parameters for processing the dataset.

    Attributes:
        data (Dataset): The original dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing the data.
        config (dict): Configuration parameters for processing the dataset.
        tokenized_data (Dataset): The tokenized dataset after applying the mapping function.

    Methods:
        __len__(): Returns the length of the tokenized dataset.
        __getitem__(item): Returns the tokenized data at the specified index.
    """

    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        logger.info("Processing data for Extractive QA")
        mapping_function = partial(_prepare_dataset, tokenizer=self.tokenizer, config=self.config)
        self.tokenized_data = self.data.map(
            mapping_function,
            batched=True,
            remove_columns=self.data.column_names,
        )

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, item):
        return self.tokenized_data[item]
