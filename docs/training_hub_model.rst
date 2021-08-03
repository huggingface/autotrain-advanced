Training A Model From Hugging Face Hub
==============================================

Using AutoNLP you can also finetune a model that is hosted on Hugging Face Hub. You can choose from one of the
10K+ models hosted here: http://hf.co/models. The model must have it's own tokenizer!

To train a model of your choice from hub, all you need to do is specify `--hub_model` parameter while creating a project.

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
        <td class="tg-0pky">0.1</td>
    </tr>
    <tr>
        <td class="tg-0pky">i dont like this movie</td>
        <td class="tg-0pky">0.5</td>
    </tr>
    <tr>
        <td class="tg-0pky">this is the best tutorial ever</td>
        <td class="tg-0pky">-1.5</td>
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

    $ autonlp create_project --name hub_model_training --task single_column_regression --hub_model abhishek/my_awesome_model --max_models 25

The hub model, "abhishek/my_awesome_model" must consist of a tokenizer and must be compatible with Hugging Face's transformers.
You can also specify "--max_models" parameter to train different variations of the same model.
When you specify "--hub_model", the language parameter is ignored and AutoNLP does everything but model search.

Everything else remains the same as any other task!
