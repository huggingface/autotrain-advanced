from ..utils import BOLD_TAG as BLD
from ..utils import CYAN_TAG as CYN
from ..utils import GREEN_TAG as GRN
from ..utils import RESET_TAG as RST


COL_MAPPING_ARG_HELP = f"""\
The files' column mapping. Must be like this:
'{GRN}col_name{RST}:{CYN}autonlp_col_name{RST},{GRN}col_name{RST}:{CYN}autonlp_col_name{RST}'
where '{CYN}autonlp_col_name{RST}' corresponds to an expected column in AutoNLP, and
'{GRN}col_name{RST}' is the corresponding column in your files.
"""

COL_MAPPING_HELP = f"""\
Expected columns for AutoNLP evaluation tasks:
--------------------------------------------------------

{BLD}col_name1{RST} and {BLD}col_name2{RST} refer to columns in your files.

{BLD}`binary_classification`{RST}:
    {BLD}col_name1{RST} -> {BLD}text{RST}    (The text to classify)
    {BLD}col_name2{RST} -> {BLD}target{RST}  (The label)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}text{RST},{GRN}col_name2{RST}:{CYN}target{RST}'

{BLD}`multi_class_classification`{RST}:
    {BLD}col_name1{RST} -> {BLD}text{RST}    (The text to classify)
    {BLD}col_name2{RST} -> {BLD}target{RST}  (The label)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}text,{GRN}col_name2{RST}:{CYN}target{RST}'

{BLD}`entity_extraction`{RST}:
    {BLD}col_name1{RST} -> {BLD}tokens{RST}  (The tokens to tag)
    {BLD}col_name2{RST} -> {BLD}tags{RST}    (The associated tags)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}tokens{RST},{GRN}col_name2{RST}:{CYN}tags{RST}'

{BLD}`speech_recognition`{RST}:
    {BLD}col_name1{RST} -> {BLD}path{RST}  (The path to the audio file, only the file name matters)
    {BLD}col_name2{RST} -> {BLD}text{RST}  (The matching speech transcription)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}path{RST},{GRN}col_name2{RST}:{CYN}text{RST}'

{BLD}`summarization`{RST}:
    {BLD}col_name1{RST} -> {BLD}text{RST}    (The text to summarize)
    {BLD}col_name2{RST} -> {BLD}target{RST}  (The summarization)
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}text{RST},{GRN}col_name2{RST}:{CYN}target{RST}'

{BLD}`extractive_question_answering`{RST}:
    {BLD}col_name1{RST} -> {BLD}context{RST}    (The context text)
    {BLD}col_name2{RST} -> {BLD}question{RST}  (The question text)
    {BLD}col_name3{RST} -> {BLD}answers.answer_start{RST}  (Character indices for the start of answers [a list of ints])
    {BLD}col_name2{RST} -> {BLD}answers.text{RST}  (Actual answer texts [a list of strings])
    Example col_mapping: --col_mapping '{GRN}col_name1{RST}:{CYN}context{RST},{GRN}col_name2{RST}:{CYN}question{RST},{GRN}col_name2{RST}:{CYN}answers.answer_start{RST},{GRN}col_name2{RST}:{CYN}answers.text{RST}'

"""
