# AutoNLP

AutoNLP: faster and easier training and deployments of NLP models

It's very easy to start a new project and train your models with AutoNLP! 


### Step 0

Install AutoNLP python package

    pip install .


### Step 1

Log into autonlp using your hugging face credentials

    $ autonlp login --username USERNAME

You can skip this step if you have `~/.autonlp/autonlp.json` file. Please use `chmod 600` on this file after logging in.

The file looks like the following:

    $ cat ~/.autonlp/autonlp.json
    {"username": "abhishek", "token": "TEST_API_KEY"}

### Step 2

Create a project. If the project is already created, it will be reused.

    $ autonlp create_project --name test_proj --task binary_classification

Valid task types are:

    binary_classification
    multi_class_classification
    multi_label_classification
    entity_extraction
    question_answering
    translation
    multiple_choice
    summarization
    lm_training

### Step 3

Upload training files to your project

    $ autonlp upload --project test_proj --split train --col_mapping review:text,sentiment:target --files ~/datasets/imdb_train.csv

Please note that you need to provide col_mapping. And similarly for validation file

    $ autonlp upload --project test_proj --split valid --col_mapping review:text,sentiment:target --files ~/datasets/imdb_valid.csv


### Step 4

After everything is done and you don't have anything errors, you are ready to train

    $ autonlp train --project test_project

Now, sit-back, relax and let the magic begin ;)
