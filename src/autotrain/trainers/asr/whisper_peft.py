from peft import PeftModel

class WhisperPeftModel(PeftModel):
    """Custom PEFT model for Whisper that handles input formatting correctly."""
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that ensures only the expected inputs are passed to the model.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The model outputs
        """
        # Filter out unexpected inputs
        whisper_kwargs = {}
        if "input_features" in kwargs:
            whisper_kwargs["input_features"] = kwargs["input_features"]
        if "labels" in kwargs:
            whisper_kwargs["labels"] = kwargs["labels"]
        
        # Call the model with only the expected inputs
        return self.model.forward(**whisper_kwargs) 