import torch


class TextClassificationDataset:
    def __init__(self, data, tokenizer, config, device=None):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.column_mapping_text_column = self.config.column_mapping_text_column
        self.column_mapping_target_column = self.config.column_mapping_target_column
        self.device = torch.device("hpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item][self.column_mapping_text_column])
        target = self.data[item][self.column_mapping_target_column]
        target = int(target)
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
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
                "input_ids": torch.tensor(ids, dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor(mask, dtype=torch.long, device=self.device),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long, device=self.device),
                "labels": torch.tensor(target, dtype=torch.long, device=self.device),
            }
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(mask, dtype=torch.long, device=self.device),
            "labels": torch.tensor(target, dtype=torch.long, device=self.device),
        }
