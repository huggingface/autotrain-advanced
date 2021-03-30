Entity Extraction
===================================

Entity extraction aka token classification is one of the most popular tasks in NLP and is fully supported in AutoNLP!

To train an entity extraction model, you need to format your data in JSONL format. It should look like the following:

.. code-block:: text

    {"tokens": ["I", "love", "AutoNLP", "!"], "labels": ["PRON", "VERB", "OBJ", "PUNCT"]}
    {"tokens": ["Hello", "there", "!"], "labels": ["EXPR", "LOC", "PUNCT"]}

Please note that both training and validation files should have the format specified above. Instead of "tokens" and "labels", you can choose whatever name you want for columns.

Once you have the data in the format specified above, you are ready to train models using AutoNLP. Yes, it's that easy.

The first step would be login to AutoNLP:

.. code-block:: bash

   $ autonlp login --api-key YOUR_HUGGING_FACE_API_TOKEN

If you do not know your Hugging Face API token, please create an account on huggingface.co and you will find your api key in settings. 
Please do not share your api key with anyone!

Once you have logged in, you can create a new project:

.. code-block:: bash

    $ autonlp create_project --name entity_model --language en --task entity_extraction

During creation of project, you can choose the language using "--language" parameter.

The next step is to upload files. Here, column mapping is very important. The columns from original data are mapped to AutoNLP column names.
In the data above, the original columns are "tokens" and "labels". We do not need more columns for a entity extraction task problem.

AutoNLP columns for entity extraction task are:

- tokens
- tags

The original columns, thus, need to be mapped to text and target. This is done in upload command. You also need to tell AutoNLP what kind of split you are uploading: train or valid.

.. code-block:: bash

    autonlp upload --project entity_model --split train \
                --col_mapping sentence:text,label:target \
                --files ~/datasets/train.csv


Similarly, upload the validation file:

.. code-block:: bash

    autonlp upload --project entity_model --split valid \
                --col_mapping sentence:text,label:target \
                --files ~/datasets/valid.csv


Please note that you can upload multiple files by separating the paths by a comma, however, the column names must be the same in each file.


Once you have uploaded the files successfully, you can start training by using the train command:

.. code-block:: bash

    $ autonlp train --project entity_model


And that's it!

Your model will start training and you can monitor the training if you wish.