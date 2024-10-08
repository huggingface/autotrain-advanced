import torch


class TextRegressionDataset:
    """
    A custom dataset class for text regression tasks for AutoTrain.

    Args:
        data (list of dict): The dataset containing text and target values.
        tokenizer (PreTrainedTokenizer): The tokenizer to preprocess the text data.
        config (object): Configuration object containing dataset parameters.

    Attributes:
        data (list of dict): The dataset containing text and target values.
        tokenizer (PreTrainedTokenizer): The tokenizer to preprocess the text data.
        config (object): Configuration object containing dataset parameters.
        text_column (str): The column name for text data in the dataset.
        target_column (str): The column name for target values in the dataset.
        max_len (int): The maximum sequence length for tokenized inputs.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(item): Returns a dictionary containing tokenized inputs and target value for a given index.
    """

    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.text_column = self.config.text_column
        self.target_column = self.config.target_column
        self.max_len = self.config.max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item][self.text_column])
        target = float(self.data[item][self.target_column])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        if "token_type_ids" in inputs:
            token_type_ids = inputs["token_type_ids"]
        else:
            token_type_ids = None

        if token_type_ids is not None:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "labels": torch.tensor(target, dtype=torch.float),
            }
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(target, dtype=torch.float),
        }
