from PIL import Image
import requests
from io import BytesIO
from nltk import pos_tag, word_tokenize
import nltk
import numpy as np
import json
import os
import re
import pickle as pkl
import random
from collections import defaultdict
from rank_bm25 import BM25Okapi
from qwen_vl_utils import process_vision_info
import torch
import socket
import io
from urllib.parse import urlparse
from htmldate import find_date
import datetime

import dynamic_mm_fc.web_related.web_utils as web_utils
import sys
sys.path.append('..')
from private_info.API_keys import GOOGLE_API_KEY,GOOGLE_SEARCH_ENGINE_ID

from google.cloud import vision
client = vision.ImageAnnotatorClient()
TEXT_QUES_PROMPT="You need to answer a question according to a set of retrieved documents. "
TEXT_QUES_PROMPT+="Question: %s. Document: %s. "
TEXT_QUES_PROMPT+="If the question is not answerable according to the provided document, please answer as: No answer can be found. Start you answer as: **ANSWER:** "

from transformers import AutoProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
clip_model.eval()
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

"""
This is implemented for reverse image search
    1. Reverse search for (multiple) images
    2. Save searched urls
    3. Scraping texts from the web
    4. Rank and select the most relevant paragraph
    5. Generate an answer based on the question and paragraph
"""
from googleapiclient.discovery import build
def google_search_text(search_term, search_type, **kwargs):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    if search_type=='text':#text query for text
        res = service.cse().list(q=search_term, cx=GOOGLE_SEARCH_ENGINE_ID, **kwargs).execute() #['link']
    elif  search_type=='image': #text query for image
        res = service.cse().list(q=search_term, searchType='image', cx=GOOGLE_SEARCH_ENGINE_ID, **kwargs).execute()
    if 'items' in res:
        return res['items']#['link']
    else:
        return []
    
def detect_web(client,path,how_many_queries=50):
    """Detects web annotations given an image."""
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    #content=base64.b64decode(content.encode())
    image = vision.Image(content=content)
    response = client.web_detection(image=image, max_results=how_many_queries)
    return response.web_detection

def compare_date(conv_ref_date,url_date):
    #print (conv_ref_date,url_date,url_date<=conv_ref_date.strftime('%Y-%m-%d'))
    #if conv_ref_date is None:
    #    conv_ref_date=datetime.date.today()
    #conv_ref_date=conv_ref_date.strftime('%Y-%m-%d')
    return url_date<=conv_ref_date
    
def det_web_valid_filter(image,date):
    #image=Image.open(img_path).convert('RGB')
    web_annotations=detect_web(client,image)
    cur_raw_ris_result=[]
    for page in web_annotations.pages_with_matching_images: 
        title=page.page_title
        if page.url.endswith(".pdf") or page.url.endswith(".doc"):
            continue
        if page.full_matching_images:
            try:
                page_date=find_date(page.url)
            except:
                page_date=None
        elif page.partial_matching_images:
            try:
                page_date=find_date(page.url)
            except:
                page_date=None
        if page_date==None or compare_date(date,page_date)==False:
            continue
        
        cur_raw_ris_result.append({
            "title":title,
            "url":page.url,
            "date":page_date
        })
    return cur_raw_ris_result

def scrap_from_ris_results(valid_web_urls):
    all_texts=[]

    for ris_result in valid_web_urls:
        url=ris_result['url']
        scraped_result=web_utils.scrape(url)
        if scraped_result is None:
            scraped_result=ris_result['title']
        scraped_result+='\nThe article was published on '+ris_result['date']
        all_texts.append(
            {
                'scrape_content':scraped_result,
                'url':url
            }
        )
    return all_texts

"""
ToDo: implementing fine-grained ranking after BM25
"""
def chunk_text(string,length=128):
    chunks=[string[0+i:length+i] for i in range(0, len(string), length)]
    return chunks
    
def rank_evid_text(query, all_text, top_k=30,fine_grained=False):
    tokenized_corpus = []
    all_corpus=[]
    for scrape_text in all_text:
        text=scrape_text['scrape_content']
        if text is None or len(text)==0:
            continue
        chunks=chunk_text(text)
        all_corpus.extend(chunks)
        for chunk in chunks:
            tokenized_corpus.append(nltk.word_tokenize(chunk))
    if len(tokenized_corpus) == 0:
        return []
    bm25 = BM25Okapi(tokenized_corpus)
    s = bm25.get_scores(nltk.word_tokenize(query))
    top_n = np.argsort(s)[::-1][:top_k]
    top_related_evid = [all_corpus[i] for i in top_n]
    return top_related_evid

def gen_retrieved_input(detailed_evid):
    texts=[]

    for i in range(len(detailed_evid)):
        texts.append(("The %d-th retrieved document: %s" % (i+1,detailed_evid[i])))
    texts="\n".join(texts)
    return texts
    
def gen_answer_with_llm(detailed_evid, ques_txt,
                        answer_llm,answer_llm_name):
    retrieved_info=gen_retrieved_input(detailed_evid)
    inputs=(TEXT_QUES_PROMPT % (ques_txt,retrieved_info))
    if 'gemini' in answer_llm_name:
        inputs=[inputs]
        response = answer_llm.models.generate_content(
            model=answer_llm_name,
            contents=inputs
        )
        answer=response.text.replace("**ANSWER:**",'').strip()
    elif answer_llm_name=='llama':
        msg=inputs
        message=[
            #{"role":"system","content":"You are a helpful assistant!"},
            {"role":"user","content":msg}
        ]
        #prompt = answer_llm['pipeline'].tokenizer.apply_chat_template(
        #    message,
        #    tokenize=False,
        #    add_generation_prompt=True
        #    )
        with torch.no_grad():
            response=answer_llm['pipeline'](message,
                                            max_new_tokens=256,
                                            #do_sample=False
                                            pad_token_id = answer_llm['pipeline'].tokenizer.eos_token_id,
                                            eos_token_id= answer_llm['terminators']
            )[0]["generated_text"][-1]["content"]
        answer=response.replace("**ANSWER:**",'').strip()
    elif answer_llm_name=='qwen':
        msg=inputs
        messages=[
            {"role":"system","content":"You are a helpful assistant!"},
            {"role":"user","content":msg}
        ]
        text = answer_llm['tokenizer'].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        model_inputs = answer_llm['tokenizer']([text], return_tensors="pt").to(answer_llm['model'].device)
        generated_ids = answer_llm['model'].generate(
            **model_inputs,
            max_new_tokens=256
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = answer_llm['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
        answer=response.replace("**ANSWER:**",'').strip()
    elif answer_llm_name=='gemma':
        msg=inputs
        messages=[
                {"role":"user","content":[{'type':'text', 'text': msg}]}
            ]
        inputs= answer_llm["processor"].apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
            #, padding="max_length", max_length=4096, truncation=True 
            ).to(answer_llm['model'].device, dtype=torch.bfloat16)
        with torch.no_grad():
            generated_ids = answer_llm['model'].generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = answer_llm["processor"].batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]      
        answer=response.replace("**ANSWER:**",'').strip()
    return answer
    
def reverse_image_search(ques_txt, img_idx, 
                         claim_txt, claim_img,
                         meta_date, meta_loc,
                         answer_llm,answer_llm_name,path):
    """
    To Be Updated!!
        Currently do not consider RIS for related images!!!!
        This kind of cases are limited, though
    """
    image_lists=[]
    for idx in img_idx:
        try:
            image_lists.append(claim_img[int(idx)-1])
        except:
            print ('Edge case: Invalid image index!!!!')
            image_lists.append(claim_img[0])#models may hallucinate about the index --> exception cases

    evid_url=None
    valid_evid=[]
    for image in image_lists:
        img_name=image.split('/')[-1].split('.')[0]
        #check if det results available
        if os.path.exists(os.path.join(path,'web_det_info/reverse_image_search/urls',img_name+'.json')):
            print ('Reverse image search results already saved!',img_name)
            valid_web_urls=load_json(os.path.join(path,'web_det_info/reverse_image_search/urls',img_name+'.json'))
        else:
            """
            Here how many queries set 100 as default!!! 
            Can be changed!!!
            """
            valid_web_urls=det_web_valid_filter(image,meta_date)
            json.dump(valid_web_urls,open(os.path.join(path,'web_det_info/reverse_image_search/urls',img_name+'.json'),'w'))

        if os.path.exists(os.path.join(path,'web_det_info/reverse_image_search/scrap',img_name+'.json')):
            all_texts=load_json(os.path.join(path,'web_det_info/reverse_image_search/scrap',img_name+'.json'))
        else:
            all_texts=scrap_from_ris_results(valid_web_urls)
            json.dump(all_texts,open(os.path.join(path,'web_det_info/reverse_image_search/scrap',img_name+'.json'),'w'))   
        valid_evid.extend(all_texts)
    #rank scarped content and answer the question
    detailed_evid=rank_evid_text(ques_txt, valid_evid)
    print ('Top related textual evidence RIS:\n\t',detailed_evid)
    try:
        answer=gen_answer_with_llm(detailed_evid, ques_txt,
                                   answer_llm,answer_llm_name)
    except:
        answer='No answer could be found.'
    return answer, detailed_evid, evid_url

"""
This is implemented for general VQA
    querying for detailed image information
"""
def vqa_mllm(ques_txt, img_idx, 
             claim_txt, claim_img,
             meta_date, meta_loc,
             answer_mllm,answer_mllm_name,root_dir):
    image_lists=[]
    for idx in img_idx:
        try:
            image_lists.append(claim_img[int(idx)-1])#idx-1 as starting from 0; path info
        except:
            print ('Edge case: Invalid image index!!!!')
            image_lists.append(claim_img[0])
    if 'gemini' in answer_mllm_name:
        inputs=["Question: "+ques_txt]
        for image in image_lists:
            im=Image.open(image).convert("RGB")
            inputs.append(im)
        response = answer_mllm.models.generate_content(
            model=answer_mllm_name,
            contents=inputs
        )
        answer=response.text.strip()
    elif answer_mllm_name in ['qwen','gemma']:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: "+ques_txt}
                ],
                }
            ]
        for image in image_lists:
            messages[0]["content"].append({'type':'image','image':image})
        
        if answer_mllm_name=='qwen':
            text = answer_mllm["processor"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
                )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = answer_mllm["processor"](
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                ).to(answer_mllm['model'].device)
        elif answer_mllm_name=='gemma':
            inputs = answer_mllm["processor"].apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
                #, padding="max_length", max_length=4096, truncation=True 
            ).to(answer_mllm['model'].device, dtype=torch.bfloat16)
        with torch.no_grad():
            generated_ids = answer_mllm['model'].generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = answer_mllm["processor"].batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    elif answer_mllm_name in ['llava','llava-inter','idefics']:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: "+ques_txt}
                ],
                }
            ]
        pil_images=[]
        for image in image_lists:
            pil_images.append(Image.open(image).convert('RGB'))
            messages[0]['content'].insert(-1,{"type": "image"})
        prompts=answer_mllm["processor"].apply_chat_template(messages, add_generation_prompt=True)
        inputs = answer_mllm["processor"](images=pil_images,
                                          text=prompts, padding=True, return_tensors="pt").to(answer_mllm['model'].device)
        if answer_mllm_name=='llava-inter':
            inputs=inputs.to(torch.float16)
        
        if answer_mllm_name=='llava':
            output =answer_mllm['model'].generate(**inputs, max_new_tokens=128, 
                                                  pad_token_id=answer_mllm["processor"].tokenizer.eos_token_id,
                                                  eos_token_id=[
                                                      answer_mllm["processor"].tokenizer.eos_token_id,
                                                      answer_mllm["processor"].tokenizer.convert_tokens_to_ids("<|eot_id|>")][0])
            answer=answer_mllm["processor"].decode(output[0], 
                                                   skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False).split('[/INST]')[-1].strip()
        elif answer_mllm_name=='llava-inter':
            output =answer_mllm['model'].generate(**inputs, max_new_tokens=128, 
                                                  pad_token_id=answer_mllm["processor"].tokenizer.eos_token_id,
                                                  eos_token_id=[
                                                      answer_mllm["processor"].tokenizer.eos_token_id,
                                                      answer_mllm["processor"].tokenizer.convert_tokens_to_ids("<|endoftext|>")])
            answer=answer_mllm["processor"].decode(output[0][len(inputs.input_ids[0]):], 
                                                   skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False).strip()
        elif answer_mllm_name=='idefics':
            answer=answer_mllm["processor"].decode(output[0], 
                                                   skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False).split('Assistant: ')[-1].strip()
    elif answer_mllm_name=='paligemma':
        prompt="Question: "+ques_txt
        pil_images=[]
        for image in image_lists:
            pil_images.append(Image.open(image).convert('RGB'))
        inputs = answer_mllm["processor"](images=[pil_images], text=prompt, return_tensors="pt").to(answer_mllm['model'].device)
        output = answer_mllm['model'].generate(**inputs, max_new_tokens=128)
        answer=self.qg_model["processor"].decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ].strip()
    return answer, None, None #placeholders for detailed evidence and evidence urls

def generate_search_query(ques_txt):
    parts = word_tokenize(ques_txt.strip())
    tags = pos_tag(parts)
    keep_tags = ["CD", "JJ", "NN", "VB"]
    search_string = []
    for (token, tag) in zip(parts, tags):
        for keep_tag in keep_tags:
            if tag[1].startswith(keep_tag):
                search_string.append(token)
    gen_ques= " ".join(search_string)
    return gen_ques

def text_search_text(ques_txt, img_idx, 
                     claim_txt, claim_img,
                     meta_date, meta_loc,
                     answer_llm,answer_llm_name,path):
    """
    ###Question###
    1. Why claim for evidence retrieval rather than questions
    2. Why reduced version of claims (verbs, adjs)
    """
    search_term=generate_search_query(ques_txt)
    search_term=search_term.replace('/',' ')
    #avoid file_name too long
    if len(search_term.split(' '))>8:
        search_term=' '.join(search_term.split(' ')[:8])
    year,month,date = meta_date.split("-")
    sort_date = year + month + date
    file_name=search_term+'_'+sort_date 
    if os.path.exists(os.path.join(path,'web_det_info/text_search_text/urls',file_name+'.json')):
        valid_web_urls=load_json(os.path.join(path,'web_det_info/text_search_text/urls',file_name+'.json'))
    else:
        """
        Here how many queries set 30 as default!!! 
        Can be changed!!!
        """
        num_pages=3
        valid_web_urls=[]
        for page_num in range(num_pages):
            try:
                search_results=google_search_text(search_term, search_type='text', 
                                                  num=10,start=0 + 10 * page_num,
                                                  sort="date:r:19000101:"+sort_date,dateRestrict=None,gl="US")
            except:
                print ('####BE CAREFUL####\n\tNO SEARCHING RESULT FROM GOOGLE!!!')
                search_results=[]
            for row in search_results:
                if 'link' not in row:
                    continue
                if row['link'].endswith(".pdf") or row['link'].endswith(".doc"):
                    continue
                if 'title' in row:
                    title=row['title']
                else:
                    title=''
                valid_web_urls.append({
                    'url':row['link'],
                    'title':title
                })
        json.dump(valid_web_urls,open(os.path.join(path,'web_det_info/text_search_text/urls',file_name+'.json'),'w'))
    if os.path.exists(os.path.join(path,'web_det_info/text_search_text/scrap',file_name+'.json')):
        all_texts=load_json(os.path.join(path,'web_det_info/text_search_text/scrap',file_name+'.json'))
    else:
        all_texts=[]
        for ris_result in valid_web_urls:
            url=ris_result['url']
            scraped_result=web_utils.scrape(url)
            all_texts.append(
                {
                    'scrape_content':scraped_result,
                    'url':url
                }
            )
        json.dump(all_texts,open(os.path.join(path,'web_det_info/text_search_text/scrap',file_name+'.json'),'w'))   
    detailed_evid=rank_evid_text(ques_txt, all_texts)
    #print ('Top related textual evidence:\n\t',detailed_evid)
    answer=gen_answer_with_llm(detailed_evid, ques_txt,
                               answer_llm,answer_llm_name)
    evid_url=None
    return answer, detailed_evid, evid_url


def rank_evid_img(cur_texts,images,top_k=1):
    if len(images)==0:
        return []
    inputs = clip_processor(
        text=[cur_texts], images=[Image.open(img).convert('RGB') for img in images], return_tensors="pt", truncation=True,
        max_length=77,
        padding=True
    ).to(clip_model.device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image.squeeze()  # this is the image-text similarity score
    ranked_logits=torch.sort(logits_per_image,descending=True,dim=0)[1][:top_k]
    ranked_images=[images[idx] for idx in ranked_logits]
    return ranked_images

def gen_answer_with_mllm(detailed_evid, ques_txt,
                         answer_mllm,answer_mllm_name):
    if 'gemini' in answer_mllm_name:
        inputs=["Question: "+ques_txt +' Please select the most relevant images from the image list and answer with the image index (1, 2, 3, etc) without any other words. Image indexing starts from 1.']
        for image in detailed_evid:
            im=Image.open(image).convert("RGB")
            inputs.append(im)
        response = answer_mllm.models.generate_content(
            model=answer_mllm_name,
            contents=inputs
        )
        gen_ans=response.text.strip()
        try:
            answer=detailed_evid[int(answer)-1]
        except:
            print ('Invalid index of image answer!')
            answer=detailed_evid[0]
    elif answer_mllm_name in ['qwen','gemma']:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: "+ques_txt +' Please select the most relevant images from the image list and answer with the image index (1, 2, 3, etc) without any other words. Image indexing starts from 1.'}
                ],
                }
            ]
        print ('Number of images:',len(detailed_evid))
        for image in detailed_evid:
            messages[0]["content"].append({'type':'image','image':image})
        if answer_mllm_name=='qwen':
            text = answer_mllm["processor"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, padding="max_length", max_length=4096, truncation=True 
                )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = answer_mllm["processor"](
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                ).to(answer_mllm['model'].device)
        elif answer_mllm_name=='gemma':
            inputs= answer_mllm["processor"].apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
                #, padding="max_length", max_length=4096, truncation=True 
                ).to(answer_mllm['model'].device, dtype=torch.bfloat16)
        with torch.no_grad():
            generated_ids = answer_mllm['model'].generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = answer_mllm["processor"].batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        try:
            answer=detailed_evid[int(answer)-1]
        except:
            print ('Invalid index of image answer!')
            answer=detailed_evid[0]
    elif answer_mllm_name in ['llava','llava-inter','idefics']:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Question: "+ques_txt +' Please select the most relevant images from the image list and answer with the image index (1, 2, 3, etc) without any other words. Image indexing starts from 1.'}
                ],
                }
            ]
        print ('Number of images:',len(detailed_evid))
        image_list=[]
        for image in detailed_evid:
            messages[0]['content'].insert(-1,{"type": "image"})
            image_list.append(Image.open(image).convert('RGB'))
        text = answer_mllm["processor"].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
            )
        prompts=answer_mllm["processor"].apply_chat_template(messages, add_generation_prompt=True)
        inputs = answer_mllm["processor"](images=image_list,
                                          text=prompts, padding=True, return_tensors="pt").to(answer_mllm['model'].device)
        if answer_mllm_name=='llava-inter':
            inputs=inputs.to(torch.float16)
        
        if answer_mllm_name=='llava':
            output =answer_mllm['model'].generate(**inputs, max_new_tokens=128,
                                                  eos_token_id=[
                                                      answer_mllm["processor"].tokenizer.eos_token_id,
                                                      answer_mllm["processor"].tokenizer.convert_tokens_to_ids("<|eot_id|>")][0])
            answer=answer_mllm["processor"].decode(output[0], 
                                                   skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False).split('[/INST]')[-1].strip()
        elif answer_mllm_name=='llava-inter':
            output =answer_mllm['model'].generate(**inputs, max_new_tokens=128,
                                                  eos_token_id=[
                                                      answer_mllm["processor"].tokenizer.eos_token_id,
                                                      answer_mllm["processor"].tokenizer.convert_tokens_to_ids("<|endoftext|>")])
            answer=answer_mllm["processor"].decode(output[0][len(inputs.input_ids[0]):], 
                                                   skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False).strip()
        elif answer_mllm_name=='idefics':
            answer=answer_mllm["processor"].decode(output[0], 
                                                   skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False).split('Assistant: ')[-1].strip()
        try:
            answer=detailed_evid[int(answer)-1]
        except:
            print ('Invalid index of image answer!')
            answer=detailed_evid[0]
    elif answer_mllm_name=='paligemma':
        prompt="Question: "+ques_txt +' Please select the most relevant images from the image list and answer with the image index (1, 2, 3, etc) without any other words. Image indexing starts from 1.'
        image_list=[]
        for image in detailed_evid:
            image_list.append(Image.open(image).convert('RGB'))
        inputs = answer_mllm["processor"](images=[image_list], text=prompt, return_tensors="pt").to(answer_mllm['model'].device)
        output = answer_mllm['model'].generate(**inputs, max_new_tokens=128,
                                               pad_token_id=answer_mllm["processor"].tokenizer.eos_token_id)
        answer=answer_mllm["processor"].decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ].strip()
        try:
            answer=detailed_evid[int(answer)-1]
        except:
            print ('Invalid index of image answer!')
            answer=detailed_evid[0]
    return answer

def text_search_image(ques_txt, img_idx, 
                      claim_txt, claim_img,
                      meta_date, meta_loc,
                      answer_mllm,answer_mllm_name,path):
    """
    1. Get searched urls and save
    2. Download images
    """
    search_term=generate_search_query(ques_txt)
    year,month,date = meta_date.split("-")
    sort_date = year + month + date
    file_name=search_term+'_'+sort_date 
    if os.path.exists(os.path.join(path,'web_det_info/text_search_image/urls',file_name+'.json')):
        valid_web_urls=load_json(os.path.join(path,'web_det_info/text_search_image/urls',file_name+'.json'))
    else:
        """
        Here how many queries set 30 as default!!! 
        Can be changed!!!
        """
        num_pages=3  #set as default
        valid_web_urls=[]
        for page_num in range(num_pages):
            try:
                search_results=google_search_text(search_term, search_type='image', 
                                                  num=10,start=0 + 10 * page_num,
                                                  sort="date:r:19000101:"+sort_date,dateRestrict=None,gl="US")
            except:
                print ('####BE CAREFUL####\n\tNO SEARCHING RESULT FROM GOOGLE!!!')
                search_results=[]

            for row in search_results:
                if 'link' not in row:
                    continue
                if 'title' in row:
                    title=row['title']
                else:
                    title=''
                valid_web_urls.append({
                    'url':row['link'],
                    'title':title
                })
        json.dump(valid_web_urls,open(os.path.join(path,'web_det_info/text_search_image/urls',file_name+'.json'),'w'))
    #downloading images ==>search_term+str(i)
    if os.path.exists(os.path.join(path,'web_det_info/text_search_image/images',file_name)):
        image_names=os.listdir(os.path.join(path,'web_det_info/text_search_image/images',file_name))
        images=[os.path.join(os.path.join(path,'web_det_info/text_search_image/images',file_name,img)) for img in image_names]
    else:
        os.mkdir(os.path.join(path,'web_det_info/text_search_image/images',file_name))
        images=[]
        for i, row in enumerate(valid_web_urls):
            url=row['url']
            #print ('Answer in the format of images:',url)
            try:
                im = Image.open(requests.get(url, stream=True).raw).convert('RGB')
                im.save(os.path.join(os.path.join(path,'web_det_info/text_search_image/images',file_name,str(i)+'.jpg')))
                images.append(os.path.join(os.path.join(path,'web_det_info/text_search_image/images',file_name,str(i)+'.jpg')))
            except:
                print ('\tError when saving the image from the url')
                continue
    top_images=rank_evid_img(ques_txt,images)
    detailed_evid=top_images
    evid_url=None
    if len(detailed_evid)==0:
        return "", detailed_evid, evid_url
    else:
        answer=gen_answer_with_mllm(detailed_evid, ques_txt,
                                    answer_mllm,answer_mllm_name)
    return answer, detailed_evid, evid_url