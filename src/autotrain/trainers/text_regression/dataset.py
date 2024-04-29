import torch


class TextRegressionDataset:
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
