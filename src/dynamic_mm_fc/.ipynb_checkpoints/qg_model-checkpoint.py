from dynamic_mm_fc.templates import ques_gen
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
import nltk
import numpy as np

from num2words import num2words

from dynamic_mm_fc.conv_utils import qa_to_evidence

class QG_Model():
    def __init__(self,qg_model,qg_model_name, 
                 answer_llm, answer_llm_name,path,
                 num_demos,
                 icl_selection_type, train_data=None):
        self.qg_model=qg_model
        self.qg_model_name=qg_model_name

        self.answer_llm=answer_llm
        self.answer_llm_name=answer_llm_name
        self.path=path #root_path
        self.num_demos=num_demos
        
        self.init_temp=ques_gen.init_ques_gen_prompt
        self.follow_temp=ques_gen.follow_ques_gen_prompt
        self.icl_temp=ques_gen.icl_ques_gen_prompt
        self.icl_selection_type=icl_selection_type
        self.para_ques_prompt=ques_gen.para_ques_prompt
        self.para_ques_prompt_icl=ques_gen.para_ques_prompt_icl

        self.train_data=train_data

        if train_data is not None:
            """
            Advanced need to save: 1) claim text/image features; 2) question features
            """
            print ('Initializing ICL selection...')
            if icl_selection_type=='basic':
                
                from rank_bm25 import BM25Okapi
                all_data_corpus = []
                tokenized_corpus = []
                for row in train_data:
                    claim_txt=row['claim_text'].strip()
                    
                    questions=row['questions']
                    if len(questions)<2:
                        continue
                    tokenized_corpus.append(nltk.word_tokenize(claim_txt))
                    question_info_pro=[]
                    for ques in questions:
                        answers=ques['answers']
                        ques_txt=ques['question'].strip()
                        """
                        Considering the computational cost, for ICL demonstrations
                            We only use the image index rather than the exact images
                        """
                        ans_info=[]
                        for ans in answers:
                            if ans['answer_type']=='Image':
                                image_answers=ans["image_answers"]
                                ans_txt=", ".join(['[IMG_'+str(3+k)+']' for k in range(len(image_answers))])
                            elif ans['answer_type']=='Boolean':
                                boolean_explanation=ans["boolean_explanation"].strip()
                                ans_txt=ans['answer_text'].strip()+' '+boolean_explanation
                            else:
                                ans_txt=ans['answer_text'].strip()
                            ans_info.append(ans_txt)
                        ans_info=' '.join(ans_info)
                        if 'Image-related' in ques['question_type'] and len(ques['input_images'])>0:
                            q_t="**Image-related:**"
                            img_idx="**Image Index:** "+",".join([str(i+1) for i,_ in enumerate(ques['input_images'])])+'.'
                        else:
                            q_t="**Text-related:**"
                            img_idx=None
                        question_info_pro.append({
                            'question':ques_txt,
                            'ques_type':q_t,
                            'answers':ans_info,
                            'img_idx':img_idx
                        })
                    all_data_corpus.append({
                        'claim':claim_txt,
                        'ques_info':question_info_pro
                    })
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.all_icl_corpus=all_data_corpus
            else:
                print('TBD!!!! No implementation yet.')

    def gen_initial_ques(self, claim_txt, claim_img):        
        if 'gemini' in self.qg_model_name:
            inputs=[(self.init_temp['first'] % claim_txt)]
            for img in claim_img:
                im=Image.open(img)
                inputs.append(im)
            inputs.append(self.init_temp['second'])
            response = self.qg_model.models.generate_content(
                model=self.qg_model_name,
                contents=inputs
            )
            ques=response.text
        elif self.qg_model_name in ['qwen','gemma']:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (self.init_temp['first'] % claim_txt)}
                    ],
                    }
                ]
            for img in claim_img:
                messages[0]["content"].append({'type':'image','image':img})
            messages[0]["content"].append({"type": "text", "text": self.init_temp['second']})
            if self.qg_model_name=='qwen':
                text = self.qg_model["processor"].apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                    )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.qg_model["processor"](
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    ).to(self.qg_model['model'].device)
            elif self.qg_model_name=='gemma':
                inputs = self.qg_model["processor"].apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                    #, padding="max_length", max_length=4096, truncation=True 
                ).to(self.qg_model['model'].device, dtype=torch.bfloat16)
                #print (inputs, messages)
            
            generated_ids = self.qg_model['model'].generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            ques = self.qg_model["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        elif self.qg_model_name in ['llava','llava-inter','idefics']:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (self.init_temp['first'] % claim_txt)}
                    ],
                    }
                ]
            image_list=[]
            for img in claim_img:
                image_list.append(Image.open(img).convert('RGB'))
                messages[0]['content'].insert(-1,{"type": "image"})
            messages[0]["content"].append({"type": "text", "text": self.init_temp['second']})
            prompts=self.qg_model["processor"].apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.qg_model["processor"](images=image_list,
                                                text=prompts, padding=True, return_tensors="pt").to(self.qg_model['model'].device)
            if self.qg_model_name=='llava-inter':
                inputs=inputs.to(torch.float16)
            
            if self.qg_model_name =='llava':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=128, 
                                                        pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                        eos_token_id=[
                                                            self.qg_model["processor"].tokenizer.eos_token_id,
                                                            self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|eot_id|>")][0],
                                                        do_sample=False)
                ques=self.qg_model["processor"].decode(output[0], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).split('[/INST]')[-1].strip()
            elif self.qg_model_name =='llava-inter':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=128, 
                                                        pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                        eos_token_id=[
                                                            self.qg_model["processor"].tokenizer.eos_token_id,
                                                            self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|endoftext|>")],
                                                        do_sample=False)
                ques=self.qg_model["processor"].decode(output[0][len(inputs.input_ids[0]):], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).strip()
            elif self.qg_model_name =='idefics':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=128, 
                                                        do_sample=True,  temperature=0.5)
                ques=self.qg_model["processor"].decode(output[0], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).split('Assistant: ')[-1].strip()
        
        elif self.qg_model_name=='paligemma':
            """
            To be updated...
            """
            prompt=(self.init_temp % claim_txt +"The image index should be smaller than "+str(len(claim_img)))
            image_list=[]
            """
            for img in claim_img:
                im=Image.open(img)
                image_list.append(im)
            """
            image_list.append(Image.open(claim_img[0]))
            inputs = self.qg_model["processor"](images=image_list, text=prompt, return_tensors="pt"
                                               ).to(torch.bfloat16).to(self.qg_model['model'].device)
            output = self.qg_model['model'].generate(**inputs, max_new_tokens=128, 
                                                     do_sample=True,  temperature=0.5)
            ques=self.qg_model["processor"].decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ]
            #[inputs.input_ids.shape[1]: ]
            print (ques,len(output[0]),inputs.input_ids.shape[1])
        """
        if self.qg_model_name in ['llava','qwen']:
            for message in messages[0]['content']:
                print (message['type'])
                if message['type']=='text':
                    print ('\t',message['text'])
                else:
                    print ('\t[IMG] ')
        """
        return ques

    def gen_follow_ques(self, claim_txt, claim_img,
                        evid_context,
                        meta_date, meta_loc,num_orig_claim_imgs
                       ): 
        if 'gemini' in self.qg_model_name:
            inputs=[(self.follow_temp['first'] % claim_txt)]
            for img in claim_img[:num_orig_claim_imgs]:
                im=Image.open(img)
                inputs.append(im)
            inputs.append((self.follow_temp['second'] % ' '.join([evid['text'] for evid in evid_context])))
            response = self.qg_model.models.generate_content(
                model=self.qg_model_name,
                contents=inputs
            )
            ques=response.text
        elif self.qg_model_name in ['qwen','gemma']:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (self.follow_temp['first'] % claim_txt)}
                    ]
                    }
                ]
            for img in claim_img[:num_orig_claim_imgs]:
                messages[0]["content"].append({'type':'image','image':img})
            messages[0]["content"].append({
                'type':'text',
                'text': (self.follow_temp['second'] % ' '.join([evid['text'] for evid in evid_context]))
            })
            if self.qg_model_name=='qwen':
                text = self.qg_model["processor"].apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                    )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.qg_model["processor"](
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    ).to(self.qg_model['model'].device)
            elif self.qg_model_name=='gemma':
                inputs= self.qg_model["processor"].apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                    #, padding="max_length", max_length=4096, truncation=True 
                    ).to(self.qg_model['model'].device, dtype=torch.bfloat16)
            generated_ids = self.qg_model['model'].generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            ques = self.qg_model["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        elif self.qg_model_name in ['llava','llava-inter','idefics']:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (self.follow_temp['first'] % claim_txt)}
                    ],
                    }
                ]
            image_list=[]
            for img in claim_img[:num_orig_claim_imgs]:
                image_list.append(Image.open(img).convert('RGB'))
                messages[0]['content'].insert(-1,{"type": "image"})
            messages[0]['content'].append({
                'type':'text',
                'text': (self.follow_temp['second'] % ' '.join([evid['text'] for evid in evid_context]))
            })
            prompts=self.qg_model["processor"].apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.qg_model["processor"](images=image_list,
                                                text=prompts, padding=True, return_tensors="pt").to(self.qg_model['model'].device)
            if self.qg_model_name=='llava-inter':
                inputs=inputs.to(torch.float16)
            
            if self.qg_model_name =='llava':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=128, 
                                                        pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                        eos_token_id=[
                                                            self.qg_model["processor"].tokenizer.eos_token_id,
                                                            self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|eot_id|>")][0], 
                                                        do_sample=False)
                ques=self.qg_model["processor"].decode(output[0], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).split('[/INST]')[-1].strip()
            elif self.qg_model_name =='llava-inter':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=128, 
                                                        pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                        eos_token_id=[
                                                            self.qg_model["processor"].tokenizer.eos_token_id,
                                                            self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|endoftext|>")],
                                                        do_sample=False)
                ques=self.qg_model["processor"].decode(output[0][len(inputs.input_ids[0]):], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).strip()
            elif self.qg_model_name =='idefics':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=128, 
                                                        do_sample=True,  temperature=0.5)
                ques=self.qg_model["processor"].decode(output[0], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).split('Assistant: ')[-1].strip()
            """
            Non-interleaved model may not prefectly following instructions in question generation
            """
            #ques=ques.split('\n')[0]
        elif self.qg_model_name=='paligemma':
            prompt=(self.follow_temp % (claim_txt, ' '.join([evid['text'] for evid in evid_context])))
            image_list=[]
            for img in claim_img[:num_orig_claim_imgs]:
                im=Image.open(img)
                image_list.append(im)
            inputs = self.qg_model["processor"](images=[image_list], text=prompt, return_tensors="pt").to(self.qg_model['model'].device)
            output = self.qg_model['model'].generate(**inputs, max_new_tokens=128,
                                                     do_sample=True,  temperature=0.5)
            ques=self.qg_model["processor"].decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ]
        return ques

    def generate_icl_texts(self, docs,len_evid):
        texts=[]
        for doc in docs:
            claim_txt=doc['claim']
            question_info=doc['ques_info']
            if len_evid==0:
                valid_q=[ques for ques in question_info if ques['ques_type']=='**Image-related:**']
                if len(valid_q)==0:
                    continue
                ques=valid_q[0]
                texts.append("Claim: %s Question: %s Image index: %s" % 
                             (claim_txt, ques['question'], ques['img_idx'].replace('**Image Index:** ','')))
                
            else:
                cur_info=[]
                
                for i,ques in enumerate(question_info):
                    if i>=2:
                        break
                    q_t=ques['ques_type']
                    answers=ques['answers']
                    img_idx=ques['img_idx']
                    question=ques['question']
                    if i==0:
                        if img_idx is not None:
                            question+= ", ".join(['[IMG_'+str(1+k)+']' for k in range(len(img_idx.split(',')))])
                        evid_history=qa_to_evidence.qa_to_evid(question, answers,
                                                               self.answer_llm,self.answer_llm_name,self.path)
                    elif i==1:
                        if q_t=="**Image-related:**":
                            cur_info.append("%s %s %s" % (q_t,question,img_idx))
                        else:
                            cur_info.append("%s %s" % (q_t,question))
                cur_info='; '.join(cur_info)
                texts.append("Claim: %s Evidence history: %s Generated questions: %s" % (claim_txt,evid_history,cur_info))
        texts='\n'.join(texts)
        print ('ICL QG:\n',texts)
        return texts

    def gen_ques_few_shot(self, claim_txt, claim_img,
                          evid_context,
                          meta_date, meta_loc, num_orig_claim_imgs,num_demo=2):
        if self.icl_selection_type=='basic':
            s = self.bm25.get_scores(nltk.word_tokenize(claim_txt))
            top_n = np.argsort(s)[::-1][:self.num_demos]
            docs = [self.all_icl_corpus[i] for i in top_n]
        demo_texts=self.generate_icl_texts(docs,len(evid_context))
        if 'gemini' in self.qg_model_name:
            if len(evid_context)==0:
                inputs=[(self.icl_temp['first'] % (demo_texts, claim_txt))]
            else:
                inputs=[(self.icl_temp['second'] % (demo_texts, claim_txt))]
            for img in claim_img[: num_orig_claim_imgs]:
                im=Image.open(img)
                inputs.append(im)
            if len(evid_context)==0:
                inputs.append(self.icl_temp['init_ques'])
            else:
                inputs.append(self.icl_temp['follow_ques'])
            response = self.qg_model.models.generate_content(
                model=self.qg_model_name,
                contents=inputs
            )
            ques=response.text
        elif self.qg_model_name in ['qwen','gemma']:
            if len(evid_context)==0:
                text=(self.icl_temp['first'] % (demo_texts, claim_txt))
            else:
                text=(self.icl_temp['second'] % (demo_texts, claim_txt))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text}
                    ]
                    }
                ]
            for img in claim_img[: num_orig_claim_imgs]:
                messages[0]["content"].append({'type':'image','image':img})
            if len(evid_context)==0:
                messages[0]["content"].append({
                    'type':'text',
                    'text': (self.icl_temp['init_ques'])
                })
            else:
                messages[0]["content"].append({
                    'type':'text',
                    'text': (self.icl_temp['follow_ques'] )
                })
            if self.qg_model_name=='qwen':
                text = self.qg_model["processor"].apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                    )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.qg_model["processor"](
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    ).to(self.qg_model['model'].device)
            elif self.qg_model_name=='gemma':
                inputs= self.qg_model["processor"].apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                    #, padding="max_length", max_length=4096, truncation=True 
                    ).to(self.qg_model['model'].device, dtype=torch.bfloat16)
            generated_ids = self.qg_model['model'].generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            ques = self.qg_model["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        elif self.qg_model_name in ['llava','llava-inter','idefics']:
            if len(evid_context)==0:
                text=(self.icl_temp['first'] % (demo_texts, claim_txt))
            else:
                text=(self.icl_temp['second'] % (demo_texts, claim_txt))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text}
                    ],
                    }
                ]
            image_list=[]
            for img in claim_img[: num_orig_claim_imgs]:
                image_list.append(Image.open(img).convert('RGB'))
                messages[0]['content'].insert(-1,{"type": "image"})
            if len(evid_context)==0:
                messages[0]["content"].append({
                    'type':'text',
                    'text': self.icl_temp['init_ques']
                })
            else:
                messages[0]["content"].append({
                    'type':'text',
                    'text': (self.icl_temp['follow_ques'])
                })
            prompts=self.qg_model["processor"].apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.qg_model["processor"](images=image_list,
                                                text=prompts, padding=True, return_tensors="pt").to(self.qg_model['model'].device)
            if self.qg_model_name=='llava-inter':
                inputs=inputs.to(torch.float16)
            
            if self.qg_model_name =='llava':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=512, 
                                                        pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                        eos_token_id=[
                                                            self.qg_model["processor"].tokenizer.eos_token_id,
                                                            self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|eot_id|>")][0],
                                                        do_sample=False)
                ques=self.qg_model["processor"].decode(output[0], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).split('[/INST]')[-1].strip()
            elif self.qg_model_name =='llava-inter':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=512, 
                                                        pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                        eos_token_id=[
                                                            self.qg_model["processor"].tokenizer.eos_token_id,
                                                            self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|endoftext|>")],
                                                        do_sample=False)
                ques=self.qg_model["processor"].decode(output[0][len(inputs.input_ids[0]):], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).strip()
            elif self.qg_model_name =='idefics':
                output =self.qg_model['model'].generate(**inputs, max_new_tokens=512,
                                                        do_sample=True,  temperature=0.5)
                ques=self.qg_model["processor"].decode(output[0], 
                                                       skip_special_tokens=True, 
                                                       clean_up_tokenization_spaces=False).split('Assistant: ')[-1].strip()
        #print ('LLaVA-Inter Question:',ques)
        ques=ques.split(';')[0]
        return ques

    def gen_question(self, evid_context,
                     claim_txt, claim_img, 
                     meta_date, meta_loc, num_orig_claim_imgs, qg_icl=False):
        #Zero-shot question generation
        if qg_icl==False:
            if len(evid_context)==0:
                #flag as the first question; generate one question for image and text, respectively
                ques=self.gen_initial_ques(claim_txt, claim_img)
            else:
                ques=self.gen_follow_ques(claim_txt, claim_img,
                                          evid_context,
                                          meta_date, meta_loc, num_orig_claim_imgs)
        #Few-shot question generation
        else:
            ques=self.gen_ques_few_shot(claim_txt, claim_img,
                                        evid_context,
                                        meta_date, meta_loc, num_orig_claim_imgs)
        return ques

    def generate_icl_para(self, docs):
        texts=["We illustrate a few examples of generated questions as below:\n"]
        for doc in docs:
            question_info=doc['ques_info']
            
            #cur_info=[]
                
            for i,ques in enumerate(question_info):
                q_t=ques['ques_type']
                img_idx=ques['img_idx']
                question=ques['question']
                
                if q_t=="**Image-related:**":
                    texts.append("%s %s %s" % (q_t,question,img_idx))
                else:
                    texts.append("%s %s" % (q_t,question))
        texts='\n'.join(texts)
        return texts

    def generate_parallel_ques(self, claim_txt, claim_img,num_questions, qg_icl=False):
        if qg_icl==False:
            num_ques=num2words(num_questions)
            instruct="You are a fact-checker to ask questions to verify an image-text claim. "
            instruct+="Here is the image-text claim. The textual part is: %s. Here is the list of images of the claim: "
            instruct=(instruct%claim_txt)
            follow_up=self.para_ques_prompt %(str(num_questions),str(num_questions))
            if 'gemini' in self.qg_model_name:
                inputs=[instruct]
                for img in claim_img:
                    im=Image.open(img)
                    inputs.append(im)
                inputs.append(follow_up)
                response = self.qg_model.models.generate_content(
                    model=self.qg_model_name,
                    contents=inputs
                )
                ques=response.text
            elif self.qg_model_name in ['gemma','qwen']:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruct}
                        ],
                        }
                    ]
                for img in claim_img:
                    messages[0]["content"].append({'type':'image','image':img})
                messages[0]["content"].append({"type": "text", "text": follow_up})
                if self.qg_model_name=='qwen':
                    text = self.qg_model["processor"].apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                        )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.qg_model["processor"](
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        ).to(self.qg_model['model'].device)
                elif self.qg_model_name=='gemma':
                    inputs=self.qg_model["processor"].apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt"
                        #, padding="max_length", max_length=4096, truncation=True 
                        ).to(self.qg_model['model'].device, dtype=torch.bfloat16)
                generated_ids = self.qg_model['model'].generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                ques = self.qg_model["processor"].batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            elif self.qg_model_name in ['llava','llava-inter','idefics']:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruct}
                        ],
                        }
                    ]
                image_list=[]
                for img in claim_img:
                    image_list.append(Image.open(img).convert('RGB'))
                    messages[0]['content'].insert(-1,{"type": "image"})
                messages[0]["content"].append({"type": "text", "text": follow_up})
                prompts=self.qg_model["processor"].apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.qg_model["processor"](images=image_list,
                                                    text=prompts, padding=True, return_tensors="pt").to(self.qg_model['model'].device)
                if self.qg_model_name=='llava-inter':
                    inputs=inputs.to(torch.float16)
            
                if self.qg_model_name =='llava':
                    output =self.qg_model['model'].generate(**inputs, max_new_tokens=1024, 
                                                            pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                            eos_token_id=[
                                                                self.qg_model["processor"].tokenizer.eos_token_id,
                                                                self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|eot_id|>")][0],
                                                            do_sample=False)
                    ques=self.qg_model["processor"].decode(output[0], 
                                                           skip_special_tokens=True, 
                                                           clean_up_tokenization_spaces=False).split('[/INST]')[-1].strip()
                elif self.qg_model_name =='llava-inter':
                    output =self.qg_model['model'].generate(**inputs, max_new_tokens=1024, 
                                                            pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                            eos_token_id=[
                                                                self.qg_model["processor"].tokenizer.eos_token_id,
                                                                self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|endoftext|>")],
                                                            do_sample=False)
                    ques=self.qg_model["processor"].decode(output[0][len(inputs.input_ids[0]):], 
                                                           skip_special_tokens=True, 
                                                           clean_up_tokenization_spaces=False).strip()
                elif self.qg_model_name =='idefics':
                    output =self.qg_model['model'].generate(**inputs, max_new_tokens=1024, 
                                                            do_sample=True,  temperature=0.5)
                    ques=self.qg_model["processor"].decode(output[0], 
                                                           skip_special_tokens=True, 
                                                           clean_up_tokenization_spaces=False).split('Assistant: ')[-1].strip()
            questions=ques.split('\n')
            questions=[ques[2:].strip() for ques in questions]
        else:
            """This could be mixed"""
            if self.icl_selection_type=='basic':
                s = self.bm25.get_scores(nltk.word_tokenize(claim_txt))
                top_n = np.argsort(s)[::-1][:self.num_demos]
                docs = [self.all_icl_corpus[i] for i in top_n]
            demo_texts=self.generate_icl_para(docs)
            num_ques=num2words(num_questions)
            instruct="You are a fact-checker to ask questions to verify an image-text claim. "
            instruct+="Here is the image-text claim. The textual part is: %s. Here is the list of images of the claim: "
            instruct=(instruct%claim_txt)
            follow_up=self.para_ques_prompt %(num_ques,num_ques)
            if 'gemini' in self.qg_model_name:
                inputs=[instruct]
                for img in claim_img:
                    im=Image.open(img)
                    inputs.append(im)
                inputs.append(follow_up)
                response = self.qg_model.models.generate_content(
                    model=self.qg_model_name,
                    contents=inputs
                )
                ques=response.text.strip()
            elif self.qg_model_name in ['qwen','gemma']:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruct}
                        ],
                        }
                    ]
                for img in claim_img:
                    messages[0]["content"].append({'type':'image','image':img})
                messages[0]["content"].append({"type": "text", "text": follow_up})
                if self.qg_model_name=='qwen':
                    text = self.qg_model["processor"].apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                        )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.qg_model["processor"](
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        ).to(self.qg_model['model'].device)
                elif self.qg_model_name=='gemma':
                    inputs= self.qg_model["processor"].apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt"
                        #, padding="max_length", max_length=4096, truncation=True 
                        ).to(self.qg_model['model'].device, dtype=torch.bfloat16)
                generated_ids = self.qg_model['model'].generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                ques = self.qg_model["processor"].batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            elif self.qg_model_name in ['llava','llava-inter','idefics']:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruct}
                        ],
                        }
                    ]
                image_list=[]
                for img in claim_img:
                    image_list.append(Image.open(img).convert('RGB'))
                    messages[0]['content'].insert(-1,{"type": "image"})
                messages[0]["content"].append({"type": "text", "text": follow_up})
                prompts=self.qg_model["processor"].apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.qg_model["processor"](images=image_list,
                                                    text=prompts, padding=True, return_tensors="pt").to(self.qg_model['model'].device)
                if self.qg_model_name=='llava-inter':
                    inputs=inputs.to(torch.float16)
            
                if self.qg_model_name =='llava':
                    output =self.qg_model['model'].generate(**inputs, max_new_tokens=1024, 
                                                            pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                            eos_token_id=[
                                                                self.qg_model["processor"].tokenizer.eos_token_id,
                                                                self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|eot_id|>")][0],
                                                            do_sample=False)
                    ques=self.qg_model["processor"].decode(output[0], 
                                                           skip_special_tokens=True, 
                                                           clean_up_tokenization_spaces=False).split('[/INST]')[-1].strip()
                elif self.qg_model_name =='llava-inter':
                    output =self.qg_model['model'].generate(**inputs, max_new_tokens=1024, 
                                                            pad_token_id=self.qg_model["processor"].tokenizer.eos_token_id,
                                                            eos_token_id=[
                                                                self.qg_model["processor"].tokenizer.eos_token_id,
                                                                self.qg_model["processor"].tokenizer.convert_tokens_to_ids("<|endoftext|>")],
                                                            do_sample=False)
                    ques=self.qg_model["processor"].decode(output[0][len(inputs.input_ids[0]):], 
                                                           skip_special_tokens=True, 
                                                           clean_up_tokenization_spaces=False).strip()
                elif self.qg_model_name =='idefics':
                    output =self.qg_model['model'].generate(**inputs, max_new_tokens=1024, 
                                                            do_sample=True,  temperature=0.5)
                    ques=self.qg_model["processor"].decode(output[0], 
                                                           skip_special_tokens=True, 
                                                           clean_up_tokenization_spaces=False).split('Assistant: ')[-1].strip()

            questions=ques.split('\n')
        
        questions=[ques for ques in questions if len(ques)>0]
        #print (questions)
        return questions