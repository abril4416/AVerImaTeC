from dynamic_mm_fc.templates import plan_gen
import torch
import numpy as np
import nltk

class Plan_Model:
    def __init__(self, plan_llm, plan_llm_name, debug, icl_selection_type, train_data=None):
        self.plan_llm=plan_llm
        self.plan_llm_name=plan_llm_name

        self.debug=debug

        self.prompt_temp=plan_gen.plan_gen_prompt
        self.prompt_temp_icl=plan_gen.plan_gen_prompt_icl
        self.train_data=train_data
        self.icl_selection_type=icl_selection_type

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
                    questions=row['questions']
                    for ques in questions:
                        answer_method=ques['answer_method']
                        if answer_method in ['Metadata','Other']:
                            continue
                        ques_type=', '.join(ques['question_type'])
                        answers=ques['answers']
                        ques_txt=ques['question'].strip()
                        tokenized_corpus.append(nltk.word_tokenize(ques_txt))
                        if 'Image-related' in ques['question_type']:
                            if answer_method=='Image Analysis':
                                plan='B'
                            else:
                                plan='A'
                        else:
                            if 'Image' in [ans['answer_type'] for ans in answers]:
                                plan='D'
                            else:
                                plan='C'
                        all_data_corpus.append({
                            'question':ques_txt,
                            'plan':plan,
                            'ques_type':ques_type
                        })
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.all_icl_corpus=all_data_corpus
            else:
                print('TBD!!!! No implementation yet.')

    def gen_next_plan_zero(self, ques, ques_type):
        if 'gemini' in self.plan_llm_name:
            inputs=[(self.prompt_temp % (ques, ques_type))]
            #response = self.plan_llm.generate_content(inputs)
            response=self.plan_llm.models.generate_content(
                model=self.plan_llm_name,
                contents=inputs
            )
            response=response.text
            plan=response.strip()
        elif self.plan_llm_name=='llama':
            msg=(self.prompt_temp % (ques, ques_type))
            message=[
                #{"role":"system","content":"You are a helpful assistant!"},
                {"role":"user","content":msg}
            ]
            #prompt = self.plan_llm['pipeline'].tokenizer.apply_chat_template(
            #prompt = self.plan_llm['pipeline'](
            #    message,
            #    tokenize=False,
            #    add_generation_prompt=True
            #    )
            with torch.no_grad():
                #response=self.plan_llm['pipeline'](prompt,
                #                                   max_new_tokens=256,
                #                                   pad_token_id = self.plan_llm['pipeline'].tokenizer.eos_token_id,
                #                                   eos_token_id=self.plan_llm['terminators'])[0]["generated_text"][len(prompt):]
                response=self.plan_llm['pipeline'](
                    message,
                    max_new_tokens=256,
                    pad_token_id = self.plan_llm['pipeline'].tokenizer.eos_token_id,
                    eos_token_id=self.plan_llm['terminators']
                    #do_sample=False
                )[0]["generated_text"][-1]["content"]
            plan=response.split(':')[-1]
        elif self.plan_llm_name=='qwen':
            msg=(self.prompt_temp % (ques, ques_type))
            messages=[
                {"role":"system","content":"You are a helpful assistant!"},
                {"role":"user","content":msg}
            ]
            text = self.plan_llm['tokenizer'].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
            model_inputs = self.plan_llm['tokenizer']([text], return_tensors="pt").to(self.plan_llm['model'].device)
            generated_ids = self.plan_llm['model'].generate(
                **model_inputs,
                max_new_tokens=256
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.plan_llm['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
            plan=response.split(':')[-1]
        elif self.plan_llm_name=='gemma':
            msg=(self.prompt_temp % (ques, ques_type))
            messages=[
                {"role":"user","content":[{'type':'text', 'text': msg}]}
            ]
            inputs= self.plan_llm["processor"].apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
                #, padding="max_length", max_length=4096, truncation=True 
                ).to(self.plan_llm['model'].device)
            with torch.no_grad():
                generated_ids = self.plan_llm['model'].generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.plan_llm["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            plan=response.split(':')[-1]            
        return plan

    def generate_icl_texts(self, docs, ques, ques_type):
        texts=[self.prompt_temp_icl]
        #if self.debug:
        #    print (ques,ques_type)
        for doc in docs:
            icl_text=("Question: %s Question type: %s Tool option: %s" % 
                      (doc['question'],doc['ques_type'],doc['plan']))
            #if self.debug:
            #    print ('\t',icl_text)
            texts.append(icl_text)
        texts='\n'.join(texts)
        texts+="\n\nQuestion: %s Question type: %s Tool option:"
        texts=(texts%( ques.split(' Image index:')[0],ques_type))
        return texts

    def gen_next_plan_few(self, ques, ques_type, num_demo=10):
        if self.icl_selection_type=='basic':
            s = self.bm25.get_scores(nltk.word_tokenize(ques))
            top_n = np.argsort(s)[::-1][:num_demo]
            docs = [self.all_icl_corpus[i] for i in top_n]
        demo_texts=self.generate_icl_texts(docs, ques, ques_type)

        if 'gemini' in self.plan_llm_name:
            inputs=[demo_texts]
            response=self.plan_llm.models.generate_content(
                model=self.plan_llm_name,
                contents=inputs
            )
            response=response.text
            plan=response.strip()
        elif self.plan_llm_name=='llama':
            msg=demo_texts
            message=[
                {"role":"system","content":"You are a helpful assistant!"},
                {"role":"user","content":msg}
            ]
            prompt = self.plan_llm['pipeline'].tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
                )
            with torch.no_grad():
                response=self.plan_llm['pipeline'](prompt,
                                                   max_new_tokens=256,
                                                   pad_token_id = self.plan_llm['pipeline'].tokenizer.eos_token_id,
                                                   eos_token_id=self.plan_llm['terminators'])[0]["generated_text"][len(prompt):]
            plan=response.split(':')[-1]
        elif self.plan_llm_name in ['qwen']:
            msg=demo_texts
            messages=[
                {"role":"system","content":"You are a helpful assistant!"},
                {"role":"user","content":msg}
            ]
            text = self.plan_llm['tokenizer'].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
            model_inputs = self.plan_llm['tokenizer']([text], return_tensors="pt").to(self.plan_llm['model'].device)
            generated_ids = self.plan_llm['model'].generate(
                **model_inputs,
                max_new_tokens=256
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.plan_llm['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
            plan=response.split(':')[-1]
        elif self.plan_llm_name=='gemma':
            msg=demo_texts
            messages=[
                {"role":"user","content":[{'type':'text', 'text': msg}]}
            ]
            inputs= self.plan_llm["processor"].apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
                ).to(self.plan_llm['model'].device)
            with torch.no_grad():
                generated_ids = self.plan_llm['model'].generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.plan_llm["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            plan=response.split(':')[-1]
        return plan

    def gen_next_plan(self, ques, ques_type, tool_icl=False):
        if tool_icl==False:
            plan=self.gen_next_plan_zero(ques, ques_type)
        elif tool_icl:
            plan=self.gen_next_plan_few(ques, ques_type)
        return plan.strip()