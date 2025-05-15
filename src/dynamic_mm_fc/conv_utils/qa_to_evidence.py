import os
root_dir=os.path.abspath('../../..')
import torch
def gen_incontext_input(ques,ans,demos):
    texts=[]
    texts.append(demos)
    texts.append("[QUES]: "+ques)
    texts.append("[ANS]: "+ans)
    texts.append("[STAT]:")
    texts='\n'.join(texts)
    return texts

def qa_to_evid(ques, ans, llm,llm_name,root_dir):
    """
    Code for converting a QA pair into evidence
        Leveraging the in-context learning capability of LLMs
    """
    #loading demonstrations
    demonstrations=open(os.path.join(root_dir,"templates/qa_to_evid_demos.txt")).readlines()
    demonstrations="".join(demonstrations)
    #print ("###Demonstrations###\n\n",demonstrations)
    incontext_input=gen_incontext_input(ques,ans,demonstrations)
    if "gemini" in llm_name:
        response = llm.models.generate_content(
            model=llm_name,
            contents=incontext_input
        )
        statement=response.text
        statement=statement.replace('[STAT]:','').strip()
    elif llm_name=='llama':
        message=[
            #{"role":"system","content":"You are a helpful assistant!"},
            {"role":"user","content":incontext_input}
        ]
        #prompt = llm['pipeline'].tokenizer.apply_chat_template(
        #    message,
        #    tokenize=False,
        #    add_generation_prompt=True
        #    )
        response=llm['pipeline'](message,
                                 max_new_tokens=256,
                                 pad_token_id = llm['pipeline'].tokenizer.eos_token_id,
                                 eos_token_id= llm['terminators']
                                 )[0]["generated_text"][-1]["content"]
        #print ('##QA Evid Demo:',incontext_input)
        print ('###QA Evid:',response)
        if len(response.split(':'))>2:
            statement=response.split(':')[-1]
        elif len(response.split('[STAT]'))>2:
            statement=response.split('[STAT]')[-1]
        else:
            statement=''.join(response.split(':')[1:])
        statement=statement.replace('[STAT]:','').strip()
        #row['evid_from_qa']=statement
        #statement=statement.replace('[STAT]','').strip()
    elif llm_name=='qwen':
        messages=[
            {"role":"system","content":"You are a helpful assistant!"},
            {"role":"user","content":incontext_input}
        ]
        text = llm['tokenizer'].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        model_inputs = llm['tokenizer']([text], return_tensors="pt").to(llm['model'].device)
        generated_ids = llm['model'].generate(
            **model_inputs,
            max_new_tokens=256
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = llm['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
        #print ('##QA Evid Demo:',incontext_input)
        #print ('###QA Evid:',response)
        if len(response.split(':'))>1:
            statement=response.split(':')[-1]
        elif len(response.split('[STAT]'))>1:
            statement=response.split('[STAT]')[-1]
        else:
            statement=response
        statement=statement.replace('[STAT]:','').strip()
    elif llm_name=='gemma':
        messages=[
                {"role":"user","content":[{'type':'text', 'text': incontext_input}]}
            ]
        inputs= llm["processor"].apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
                #, padding="max_length", max_length=4096, truncation=True 
                ).to(llm['model'].device, dtype=torch.bfloat16)
        with torch.no_grad():
            generated_ids = llm['model'].generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = llm["processor"].batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]       
        #print ('##QA Evid Demo:',incontext_input)
        #print ('###QA Evid:',response)
        if len(response.split(':'))==2:
            statement=response.split(':')[-1]
        elif len(response.split('[STAT]'))==2:
            statement=response.split('[STAT]')[-1]
        else:
            statement=response
        statement=statement.replace('[STAT]:','').strip()
    return statement