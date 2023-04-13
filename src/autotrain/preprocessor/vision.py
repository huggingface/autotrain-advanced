from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import Dataset
from loguru import logger
from sklearn.model_selection import train_test_split


RESERVED_COLUMNS = ["autotrain_text", "autotrain_label"]


@dataclass
class ImageBinaryClassificationPreprocessor:
    train_data: pd.DataFrame
    image_column: str
    label_column: str
    username: str
    project_name: str
    token: str
    valid_data: Optional[pd.DataFrame] = None
    test_size: Optional[float] = 0.2
    seed: Optional[int] = 42
