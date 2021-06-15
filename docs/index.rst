🤗 AutoNLP
===================================

AutoNLP: Auto training and fast deployment for state-of-the-art NLP models

AutoNLP is an automatic way to train, evaluate and deploy state-of-the-art NLP models for different tasks. 
Using AutoNLP, you can leave all the worries of selecting the best model, fine-tuning the model or even deploying the models and
focus on the broader picture for your project/business.

Main features:
----------------------------------------------------------------------------------------------------

 - Automatic selection of best models given your data
 - Automatic fine-tuning
 - Automatic hyperparameter optimization
 - Model comparison after training
 - Immediate deployment after training
 - CLI and Python API available

Supported Tasks
----------------------------------------------------------------------------------------------------
Currently, AutoNLP supports the following tasks:

- Binary classification: one sentence has one target associated with it and there are two unique targets in the dataset
- Multi-class classification: one sentence has one target associated with it and there are more than two unique targets in the dataset
- Entity extraction: also known as named entity recognition or token classification. This task consists of one sentence and in the sentence, each token is associated to a particular label
- Summarization: a sequence to sequence task in which the larger sequence is summarized to smaller sequence
- Speech recognition: train your own automatic speech recognition model using AutoNLP
- Single-column regression: train your own regression model
- Extractive question answering: train custom question-answering models on your own dataset


.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   supported_languages
   training_hub_model
   binary_classification
   multi_class_classification
   entity_extraction
   summarization
   speech_recognition
   single_column_regression
   extractive_question_answering
