META_EVID="The claim was made on % and was made in %s."

from dynamic_mm_fc import qg_model, planner, qa_model, verifier, justification_gen
from dynamic_mm_fc.conv_utils import qa_to_evidence
import config
from dynamic_mm_fc.utils import parse_ques
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import datetime
import json
import os
import pickle as pkl
import random
import pycountry
import torch
import transformers

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

if __name__ == '__main__':
    args=config.parse_opt()
    
    llm_name=args.LLM_NAME
    mllm_name=args.MLLM_NAME
    print ('Name:',llm_name,mllm_name)
    if 'gemini' in llm_name:
        #import google.generativeai as genai
        from google import genai
        from google.genai.types import HttpOptions
        import sys
        sys.path.append('..')
        from private_info.API_keys import GEMINI_API_KEY
        llm_model = genai.Client(http_options=HttpOptions(api_version="v1"), api_key=GEMINI_API_KEY)
    elif llm_name=='llama':#using llama-3.1
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda:0",
            )
        pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        llm_model={
            'pipeline':pipeline,
            'terminators':terminators
        }
    elif llm_name=='qwen':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        llm_model={
            'model':model,
            'tokenizer':tokenizer
        }
    elif llm_name=='gemma':
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        ckpt = "google/gemma-3-12b-it"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            ckpt, device_map="cuda:0", torch_dtype=torch.bfloat16,
        ).eval()          
        model=model.bfloat16()
        processor = AutoProcessor.from_pretrained(ckpt)
        processor.tokenizer.pad_token = "[PAD]"
        processor.tokenizer.padding_side = "left"
        llm_model={
            'model':model,
            'processor':processor
        }
    print (llm_name,mllm_name)
    all_results=load_pkl(os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name]),str(args.SAVE_NUM)+'.pkl'))
    save_path=os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name]),str(args.SAVE_NUM)+'_justification.pkl')
    print ('Loading results from:','_'.join([llm_name,mllm_name])+str(args.SAVE_NUM)+'.pkl')
    print ('\tLength of results:',len(all_results))

    prompt_head="You are a helpful assistant to summarize a textual justification in one or two sentences. Here is the justification: %s. Please generate your summarization:"
    if os.path.exists(save_path):
        justifications=load_pkl(save_path)
    else:
        justifications={}
    vis=0
    for req_id in all_results:
        if vis%100==0:
            pkl.dump(justifications,open(save_path,'wb'))
            print ('\tAlready finished:',vis)
        justi=' '.join(all_results[req_id]['justification'].split('\n'))
        inputs=(prompt_head%justi)
        if 'gemini' in llm_name:
            #response = self.plan_llm.generate_content(inputs)
            response=llm_model.models.generate_content(
                model=llm_name,
                contents=[inputs]
            )
            response=response.text
            plan=response.strip()
        elif llm_name=='qwen':
            msg=inputs
            messages=[
                {"role":"system","content":"You are a helpful assistant!"},
                {"role":"user","content":msg}
            ]
            text = llm_model['tokenizer'].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
            model_inputs = llm_model['tokenizer']([text], return_tensors="pt").to(llm_model['model'].device)
            generated_ids = llm_model['model'].generate(
                **model_inputs,
                max_new_tokens=256
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = llm_model['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
            plan=response.split(':')[-1]
        elif llm_name=='gemma':
            msg=inputs
            messages=[
                {"role":"user","content":[{'type':'text', 'text': msg}]}
            ]
            inputs= llm_model["processor"].apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
                #, padding="max_length", max_length=4096, truncation=True 
                ).to(llm_model['model'].device)
            with torch.no_grad():
                generated_ids = llm_model['model'].generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = llm_model["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            plan=response.split(':')[-1]            
        justifications[req_id]=plan
        vis+=1
    pkl.dump(justifications,open(save_path,'wb'))
