# Baseline Implementation for AVerImaTeC

This repository maintains the baseline described in our paper: AVERIMATEC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web

## Content
- [Dataset Preparation](#dataset-preparation)
- [Experiment Setting](#experiment-setting)
- [Baseline Implementation](#baseline-implementation)
- [Baseline Evaluation](#baseline-evaluation)

## Dataset Preparation

### AVerImaTeC Data
Please download the data from our provided link. Put the *images.zip* under the *data/data_clean* folder and unzip it. For json files, please put it under the *data/data_clean/split_data*. 

### API Keys
In order to use Gemini and Google search, you need to put your own API keys in the folder *private_info*.

## Experiment Setting

In order to implement our baselines, you need to install essential packages listed in *requirement.txt*. 

## Baseline Implementation

## Baseline Evaluation
The evaluation consists of two parts: for the generated questions and evidence. Question evaluation follows the [Ev2R paper](https://arxiv.org/abs/2411.05375). We extend it to the multimodal setting for the evaluation of interleaved image-text evidence. After generating predictions following [the section above](#baseline-implmenetation), you can execute the script below under the *evaluation* folder, for evidence evaluation:
```
python evid_eval.py --llm_name [LLM_IN_BASELINE] --mllm_name [MLLM_IN_BASELINE] --save_num [SAVE_NUM_FOR_PREDICTION] --seperate_val True 
```
We also provide options for text-only and interleaved image-text evaluation (see SEction 6 of our paper), which have worse performance compared to the separated evaluation. You can try these evaluation methods by setting *--text-only True* or *--text-only False* + *--seperate_val False*

We also provided post-doc calculations for the evidence evaluation score, verdict prediction accuracy and justification generation: see *evaluation/ipython/Val_Evid_Latest.ipynb*.
