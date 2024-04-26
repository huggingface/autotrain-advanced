import torch


class RegressionDataset:
    def __init__(self, data, tokenizer, zeus_config, model_config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = zeus_config
        self.model_config = model_config
        self.max_len = self.config.max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item]["text"])
        target = float(self.data[item]["target"])
        inputs = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True)

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
