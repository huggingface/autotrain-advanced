<a href="https://github.com/huggingface/optimum-habana#gh-light-mode-only">
  <img src="https://github.com/huggingface/optimum-habana/blob/main/readme_logo_light.png"/>
</a>

<a href="https://github.com/huggingface/optimum-habana#gh-dark-mode-only">
  <img src="https://github.com/huggingface/optimum-habana/blob/main/readme_logo_dark.png"/>
</a>

# 🤗 AutoTrain Advanced for Intel Gaudi

AutoTrain Advanced is a no-code solution that allows you to train machine learning models in just a few clicks. We take the AutoTrain Advanced from HuggingFace and extend it for Intel's Gaudi products. Please note that you must upload data in correct format for project to be created. For help regarding proper data format and pricing, check out the documentation.

NOTE: AutoTrain is free! You only pay for the resources you use in case you decide to run AutoTrain on Hugging Face Spaces. When running locally, you only pay for the resources you use on your own infrastructure.

## What are Intel Gaudi AI Accelerators (HPUs)?

HPUs offer fast model training and inference as well as a great price-performance ratio.
Check out [this blog post about BLOOM inference](https://huggingface.co/blog/habana-gaudi-2-bloom) and [this post benchmarking Intel Gaudi 2 and NVIDIA A100 GPUs for BridgeTower training](https://huggingface.co/blog/bridgetower) for concrete examples.


## Gaudi Setup

Please refer to the Intel Gaudi AI Accelerator official [installation guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).

> Tests should be run in a Docker container based on Intel Gaudi Docker images.
>
> The current version has been validated for SynapseAI 1.16.

## Local Installation

You can Install AutoTrain-Advanced python package via PIP. Please note you will need python >= 3.10 for AutoTrain Advanced to work properly.

    pip install autotrain-advanced
    
Please make sure that you have git lfs installed. Check out the instructions here: https://github.com/git-lfs/git-lfs/wiki/Installation

You also need to install torch, torchaudio and torchvision.

The best way to run autotrain is in a conda environment. You can create a new conda environment with the following command:

    conda create -n autotrain python=3.10
    conda activate autotrain
    pip install autotrain-advanced
    conda install pytorch torchvision torchaudio

Once done, you can start the application using:

    autotrain app --port 8080 --host 127.0.0.1


If you are not fond of UI, you can use AutoTrain Configs to train using command line or simply AutoTrain CLI.

To use config file for training, you can use the following command:

    autotrain --config <path_to_config_file>


You can find sample config files in the `configs` directory of this repository.


## Documentation

Documentation is available at 
