import csv
import json
import os
from functools import partial
from typing import Dict, Iterable, TextIO


SUPPORTED_AUDIO_FILE_FORMAT = ("mp3", "wav")


def audio_file_name_iter(transcription_file_path: str, col_mapping: Dict[str, str]) -> Iterable[str]:
    filename = os.path.basename(transcription_file_path)
    file_ext = filename.split(".")[-1]
    path_column = [src for src, dst in col_mapping.items() if dst == "path"][0]

    if file_ext in ("csv", "tsv"):
        delimiter = "\t" if file_ext == "tsv" else ","
        sample_iterator = partial(csv.DictReader, delimiter=delimiter)
    else:
        sample_iterator = json_line_iterator

    with open(transcription_file_path, encoding="utf-8") as f:
        for sample in sample_iterator(f):
            path = sample[path_column]
            yield os.path.basename(path)


def json_line_iterator(f: TextIO) -> Iterable[dict]:
    for line in f:
        yield json.loads(line)
