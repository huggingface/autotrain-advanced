from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from autotrain import logger


class AutoTrainPreprocessor(ABC):
    """
    Base class for all AutoTrain preprocessors.
    """

    def __init__(
        self,
        train_data: str,
        token: str,
        project_name: str,
        username: str,
        column_mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize the preprocessor.

        Args:
            train_data (str): Path to the training data
            token (str): Hugging Face token
            project_name (str): Name of the project
            username (str): Username of the project owner
            column_mapping (Optional[Dict[str, str]]): Mapping of column names
            **kwargs: Additional arguments
        """
        self.train_data = train_data
        self.token = token
        self.project_name = project_name
        self.username = username
        self.column_mapping = column_mapping or {}
        self.kwargs = kwargs

    @abstractmethod
    def prepare(self) -> Dict[str, Any]:
        """
        Prepare the data for training.

        Returns:
            Dict[str, Any]: Dictionary containing the prepared data
        """
        pass

    def log(self, message: str):
        """
        Log a message.

        Args:
            message (str): Message to log
        """
        logger.info(message) 