Extractive Question Answering
===================================

Extractive Question Answering is a task in Natural Language Processing where you are given a context and a question.
The model's job is to to extract answer from the context.

To train Extractive Question Answering models using AutoNLP, you need your data to be in JSONL format. An example is shown below:


.. code-block:: text

    {"answers.start_idx":[515],"answers.ans_text":["Saint Bernadette Soubirous"],"context":"Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.","id":"5733be284776f41900661182","question":"To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?","title":"University_of_Notre_Dame"}
    {"answers.start_idx":[0],"answers.ans_text":["Kathmandu Metropolitan City"],"context":"Kathmandu Metropolitan City (KMC), in order to promote international relations has established an International Relations Secretariat (IRC). KMC's first international relationship was established in 1975 with the city of Eugene, Oregon, United States. This activity has been further enhanced by establishing formal relationships with 8 other cities: Motsumoto City of Japan, Rochester of the USA, Yangon (formerly Rangoon) of Myanmar, Xi'an of the People's Republic of China, Minsk of Belarus, and Pyongyang of the Democratic Republic of Korea. KMC's constant endeavor is to enhance its interaction with SAARC countries, other International agencies and many other major cities of the world to achieve better urban management and developmental programs for Kathmandu.","id":"5735d259012e2f140011a0a1","question":"What is KMC an initialism of?","title":"Kathmandu"}

Here, we see only two samples but you can have as many samples as you like: 5000, 10000, 100000 or even a million or more!

Please note that one question can have multiple answers!

Once you are done with formatting data in the format specified above, training the model is a piece of cake (thanks to AutoNLP).

The first step would be login to AutoNLP:

.. code-block:: bash

   $ autonlp login --api-key YOUR_HUGGING_FACE_API_TOKEN

If you do not know your Hugging Face API token, please create an account on huggingface.co and you will find your api key in settings. 
Please do not share your api key with anyone!

Once you have logged in, you can create a new project:

.. code-block:: bash

    $ autonlp create_project --name qa_project --language en --task extractive_question_answering

During creation of project, you can choose the language using "--language" parameter.

The next step is to upload files. Here, column mapping is very important. The columns from original data are mapped to AutoNLP column names.
In the data above, the original columns are "context", "question", "answers.start_idx" & "answers.ans_text". We do not need more columns for an extractive question answering task.

AutoNLP columns for extractive question answering are:

- context
- question
- answers.answer_start
- answers.text

The original columns, thus, need to be mapped to these columns. This is done in upload command. You also need to tell AutoNLP what kind of split you are uploading: train or valid.

.. code-block:: bash

    autonlp upload --project qa_project --split train \
                --col_mapping answers.start_idx:answers.answer_start,answers.ans_text:answers.text,context:context,question:question \
                --files ~/datasets/train.csv


Similarly, upload the validation file:

.. code-block:: bash

    autonlp upload --project qa_project --split valid \
                --col_mapping answers.start_idx:answers.answer_start,answers.ans_text:answers.text,context:context,question:question \
                --files ~/datasets/valid.csv

Column mapping is always from original column to AutoNLP column (original_column:autonlp_column).


Please note that you can upload multiple files by separating the paths by a comma, however, the column names must be the same in each file.


Once you have uploaded the files successfully, you can start training by using the train command:

.. code-block:: bash

    $ autonlp train --project qa_project


And that's it!

Your model will start training and you can monitor the training if you wish.
