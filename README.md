# 🎊 News <!-- omit in toc -->

- [2024.09] 🔥 AVerImaTeC shared task has been held [here](https://huggingface.co/spaces/FEVER-IT/AVerImaTeC). The baseline implementation with static knowledge store is provided in this [repository](https://github.com/abril4416/AVerImaTec_Shared_Task).
- [2025.09] 🎉 We are pleased to announce that FEVER9 will be co-located with EACL2026! In this year's workshop, we will focus on image-text claim verification and leverage AVerImaTeC as the shared task. You can learn more about FEVER9 and past FEVER workshops [here](https://fever.ai/index.html).
- [2025.09] 🎉 Our AVerImaTeC paper is accepted by NeurIPS Datasets and Benchmarks track! You can access the lastest version of the paper at [here](https://arxiv.org/pdf/2505.17978).

# Baseline Implementation for AVerImaTeC

This repository maintains the baseline described in our paper: AVERIMATEC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web

## Content
- [Dataset Preparation](#dataset-preparation)
- [Experiment Setting](#experiment-setting)
- [Baseline Implementation](#baseline-implementation)
- [Baseline Evaluation](#baseline-evaluation)

## Dataset Preparation

### AVerImaTeC Data
Please download the data from our provided [link](https://huggingface.co/datasets/Rui4416/AVerImaTeC). Put the *images.zip* under the *data/data_clean* folder and unzip it. For json files, please put it under the *data/data_clean/split_data*. 

### API Keys
In order to use Gemini and Google search, you need to put your own API keys in the folder *private_info*.

## Experiment Setting

In order to implement our baselines, you need to install essential packages listed in *requirement.txt*. Besides, you need to set up Google Cloud Vision for *Reverse Image Search* and Google Client for *Google Customized Search*. More details can be found [here](https://cloud.google.com/vision/docs/detecting-web) and [here](https://developers.google.com/custom-search/v1/overview).

## Baseline Implementation

For instance, for one combination of an LLM and an MLLM, you are able to get six results, three for the few-shot setting and three for the zero-shot setting. For each setting, you can obtain results with three question generation strategies. We list the six commands below to replicate results shown in Table 3 and 12 of our paper. We use Qwen for an illustration:
```
python mm_checker.py --LLM_NAME 'qwen' --MLLM_NAME 'qwen' --SAVE_NUM 11
python mm_checker.py --LLM_NAME 'qwen' --MLLM_NAME 'qwen' --SAVE_NUM 12 --QG_ICL True
python mm_checker.py --LLM_NAME 'qwen' --MLLM_NAME 'qwen' --SAVE_NUM 13 --PARA_QG True
python mm_checker.py --LLM_NAME 'qwen' --MLLM_NAME 'qwen' --SAVE_NUM 14 --PARA_QG True --QG_ICL True 
python mm_checker.py --LLM_NAME 'qwen' --MLLM_NAME 'qwen' --SAVE_NUM 15 --HYBRID_QG True --NUM_GEN_QUES 2  
python mm_checker.py --LLM_NAME 'qwen' --MLLM_NAME 'qwen' --SAVE_NUM 16 --HYBRID_QG True --NUM_GEN_QUES 2 --QG_ICL True 
```
You can also set *--DEBUG True* to switch to the debug mode (only test a few claims) for easy debugging.
We provide the implementation for the following LLMs and MLLMs: LLM_NAME can be set to *gemini-2.0-flash-001*, *qwen* and *gemma* while MLLM_NAM can be set to *gemini-2.0-flash-001*, *qwen*, *gemma* and *llava*.

As MLLMs tend to generate long outputs, we implement a post-doc summarization step, with the script:
```
python summarize_justification.py --LLM_NAME [LLM_IN_BASELINE] --MLLM_NAME [MLLM_IN_BASELINE] --SAVE_NUM [SAVE_NUM_FOR_PREDICTION]
```

## Baseline Evaluation
The evaluation consists of two parts: for the generated questions and evidence. Question evaluation follows the [Ev2R paper](https://arxiv.org/abs/2411.05375). We extend it to the multimodal setting for the evaluation of interleaved image-text evidence. After generating predictions following [the section above](#baseline-implmenetation), you can execute the script below under the *evaluation* folder, for evidence evaluation:
```
python evid_eval.py --llm_name [LLM_IN_BASELINE] --mllm_name [MLLM_IN_BASELINE] --save_num [SAVE_NUM_FOR_PREDICTION] --seperate_val True 
```
We also provide options for text-only and interleaved image-text evaluation (see Section 6 of our paper), which have worse performance compared to the separated evaluation. You can try these evaluation methods by setting *--text-only True* or *--text-only False* + *--seperate_val False*

We also provided post-doc calculations for the evidence evaluation score, verdict prediction accuracy and justification generation: see *evaluation/ipython/Val_Evid_Latest.ipynb*.
