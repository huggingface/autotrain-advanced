# 🤗 AutoTrain Advanced

AutoTrain Advanced: faster and easier training and deployments of state-of-the-art machine learning models. AutoTrain Advanced is a no-code solution that allows you to train machine learning models in just a few clicks. Please note that you must upload data in correct format for project to be created. For help regarding proper data format and pricing, check out the documentation. 

NOTE: AutoTrain is free! You only pay for the resources you use in case you decide to run AutoTrain on Hugging Face Spaces. When running locally, you only pay for the resources you use on your own infrastructure.


## Run on Colab or Hugging Face Spaces

- Run AutoTrain on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/autotrain-advanced/blob/main/colabs/AutoTrain.ipynb)


- Deploy AutoTrain on Hugging Face Spaces: [![Deploy on Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-on-spaces-md.svg)](https://huggingface.co/login?next=%2Fspaces%2Fautotrain-projects%2Fautotrain-advanced%3Fduplicate%3Dtrue)


- Run AutoTrain UI on Colab via ngrok: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/autotrain-advanced/blob/main/colabs/AutoTrain_ngrok.ipynb)


## Local Installation

You can Install AutoTrain-Advanced python package via PIP. Please note you will need python >= 3.10 for AutoTrain Advanced to work properly.

    pip install autotrain-advanced
    
Please make sure that you have git lfs installed. Check out the instructions here: https://github.com/git-lfs/git-lfs/wiki/Installation

You also need to install torch, torchaudio and torchvision.

The best way to run autotrain is in a conda environment. You can create a new conda environment with the following command:

    conda create -n autotrain python=3.10
    conda activate autotrain
    pip install autotrain-advanced
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -c "nvidia/label/cuda-12.1.0" cuda-nvcc

Once done, you can start the application using:

    autotrain app --port 8080 --host 127.0.0.1


If you are not fond of UI, you can use AutoTrain Configs to train using command line or simply AutoTrain CLI.

To use config file for training, you can use the following command:

    autotrain --config <path_to_config_file>


You can find sample config files in the `configs` directory of this repository.

## Colabs

| Task | Colab Link |
| --- | --- |
| LLM Fine Tuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/autotrain-advanced/blob/main/colabs/AutoTrain_LLM.ipynb) |
| DreamBooth Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/autotrain-advanced/blob/main/colabs/AutoTrain_Dreambooth.ipynb) |


## Documentation

Documentation is available at https://hf.co/docs/autotrain/
