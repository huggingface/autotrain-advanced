Binary Classification
===================================

Binary classification is one the most popular supervised classification problem one might come across when dealing with NLP problems.
AutoNLP makes it super easy to train binary classification models on your data. Let's assume we are training a model for sentiment detection.
The dataset has two sentiments: positive & negative.

Let's assume our data is in CSV format and looks something like the following:

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
    </style>
    <table class="tg">
    <thead>
    <tr>
        <th class="tg-0pky"><span style="font-weight:bold">sentence</span></th>
        <th class="tg-0pky"><span style="font-weight:bold">label</span></th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td class="tg-0pky">i love autonlp</td>
        <td class="tg-0pky">positive</td>
    </tr>
    <tr>
        <td class="tg-0pky">i dont like this movie</td>
        <td class="tg-0pky">negative</td>
    </tr>
    <tr>
        <td class="tg-0pky">this is the best tutorial ever</td>
        <td class="tg-0pky">negative</td>
    </tr>
    </tbody>
    </table>
    <br />

Here, we see only three samples but you can have as many samples as you like: 5000, 10000, 100000 or even a million or more!

Once you have the data in the format specified above, you are ready to train models using AutoNLP. Yes, it's that easy.

The first step would be login to AutoNLP:

.. code-block:: bash

   $ autonlp login --api-key YOUR_HUGGING_FACE_API_TOKEN

If you do not know your Hugging Face API token, please create an account on huggingface.co and you will find your api key in settings. 
Please do not share your api key with anyone!

Once you have logged in, you can create a new project:

.. code-block:: bash

    $ autonlp create_project --name sentiment_detection --language en --task binary_classification

During creation of project, you can choose the language using "--language" parameter.

The next step is to upload files. Here, column mapping is very important. The columns from original data are mapped to AutoNLP column names.
In the data above, the original columns are "sentence" and "label". We do not need more columns for a binary classification problem.

AutoNLP columns for binary classification are:

- text
- target

The original columns, thus, need to be mapped to text and target. This is done in upload command. You also need to tell AutoNLP what kind of split you are uploading: train or valid.

.. code-block:: bash

    autonlp upload --project sentiment_detection --split train \
                --col_mapping sentence:text,label:target \
                --files ~/datasets/train.csv


Similarly, upload the validation file:

.. code-block:: bash

    autonlp upload --project sentiment_detection --split valid \
                --col_mapping sentence:text,label:target \
                --files ~/datasets/valid.csv

Column mapping is always from original column to AutoNLP column (original_column:autonlp_column).

Please note that you can upload multiple files by separating the paths by a comma, however, the column names must be the same in each file.


Once you have uploaded the files successfully, you can start training by using the train command:

.. code-block:: bash

    $ autonlp train --project sentiment_detection


And that's it!

Your model will start training and you can monitor the training if you wish.