class Seq2SeqDataset:
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_len_input = self.config.max_seq_length
        self.max_len_target = self.config.max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item][self.config.text_column])
        target = str(self.data[item][self.config.target_column])

        model_inputs = self.tokenizer(text, max_length=self.max_len_input, truncation=True)

        labels = self.tokenizer(text_target=target, max_length=self.max_len_target, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
