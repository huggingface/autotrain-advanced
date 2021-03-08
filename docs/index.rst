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

Supported Languages
----------------------------------------------------------------------------------------------------
Currently, AutoNLP supports the following languages:

- English: en
- French: fr
- German: de
- Finnish: fi
- Hindi: hi
- Spanish: es
- Chinese: zh
- Dutch: nl

If the language you want to use is not listed, please create an issue here: https://github.com/huggingface/autonlp/issues and we will try our best to add the languages you need.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   binary_classification
   multi_class_classification
   entity_extraction
