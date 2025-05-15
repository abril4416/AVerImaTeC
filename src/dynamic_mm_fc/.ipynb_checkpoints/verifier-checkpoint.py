from dynamic_mm_fc.templates import verify_gen
from PIL import Image
from dynamic_mm_fc.utils import split_string_by_words
import re
from qwen_vl_utils import process_vision_info
import torch

class Verify_Model:
    def __init__(self, verify_mllm, verify_mllm_name):
        self.verify_mllm=verify_mllm
        self.verify_mllm_name=verify_mllm_name

        self.prompt_temp=verify_gen.verify_gen_prompt

    def verify(self, meta_info, claim_txt, claim_img, evid_context, num_orig_claim_imgs):
        #evid=[str(i+1)+'. '+evid for i,evid in enumerate(evid_context)]
        #evid=' '.join(evid)
        #print (claim_img)
        #print (evid_context)
        if 'gemini' in self.verify_mllm_name:
            inputs=[(self.prompt_temp % (meta_info, claim_txt))]
            for img in claim_img[:num_orig_claim_imgs]:
                inputs.append(Image.open(img).convert('RGB'))
            inputs.append("\nEvidence: ")
            for i,evid in enumerate(evid_context):
                evid_text=evid['text']
                evid_images=evid['images']
                if len(evid_images)==0:
                    inputs.append((str(i+1)+". "+evid_text))
                else:
                    img_token_list=re.findall(r"\[IMG_.*?\]",evid_text) # [IMG_1], [IMG_2]...
                    if len(img_token_list)==0:
                        inputs.append((str(i+1)+". "+evid_text))
                    else:
                        split_string=split_string_by_words(evid_text,img_token_list)
                        for m,sp_str in enumerate(split_string):
                            if sp_str in img_token_list:
                                img_idx=re.findall(r'\d+',sp_str)[0]
                                try:
                                    inputs.append(Image.open(evid_images[int(img_idx)-1]).convert('RGB'))
                                except:
                                    continue
                            else:
                                if m==0:
                                    inputs.append(str(i+1)+". "+sp_str)
                                else:
                                    inputs.append(sp_str)
            inputs.append("\nVerdict:")
            response = self.verify_mllm.models.generate_content(
                model=self.verify_mllm_name,
                contents=inputs
            )
            response=response.text
            verdict=response.strip()
        elif self.verify_mllm_name in ['qwen','gemma']:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (self.prompt_temp % (meta_info, claim_txt))}
                    ],
                    }
                ]
            num_images=len(claim_img)
            for img in claim_img[:num_orig_claim_imgs]:
                messages[0]["content"].append({'type':'image','image':img})
            already_text_to_image_search=False
            messages[0]["content"].append({'type':'text',"text": '\nHere is the evidence: '})
            for i,evid in enumerate(evid_context):
                evid_text=evid['text']
                evid_images=evid['images']
                if len(evid_images)==0:
                    messages[0]["content"].append({'type':'text','text':str(i+1)+". "+evid_text})
                else:
                    img_token_list=re.findall(r"\[IMG_.*?\]",evid_text) # [IMG_1], [IMG_2]...
                    if len(img_token_list)==0:
                        messages[0]["content"].append({'type':'text','text':str(i+1)+". "+evid_text})
                    else:
                        split_string=split_string_by_words(evid_text,img_token_list)
                        for m,sp_str in enumerate(split_string):
                            if sp_str in img_token_list:
                                img_idx=re.findall(r'\d+',sp_str)[0]
                                #if num_images<5:
                                try:
                                    messages[0]["content"].append({'type':'image','image':evid_images[int(img_idx)-1]})
                                except:
                                    continue
                                #    num_images+=1
                            else:
                                if m==0:
                                    messages[0]["content"].append({'type':'text','text':str(i+1)+". "+sp_str})
                                else:
                                    messages[0]["content"].append({'type':'text','text':sp_str})
            messages[0]["content"].append({'type':'text','text':"\nVerdict:"})
            if self.verify_mllm_name=='qwen':
                text = self.verify_mllm["processor"].apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                    )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.verify_mllm["processor"](
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    ).to(self.verify_mllm['model'].device)
            elif self.verify_mllm_name=='gemma':
                inputs= self.verify_mllm["processor"].apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                    #, padding="max_length", max_length=4096, truncation=True 
                    ).to(self.verify_mllm['model'].device, dtype=torch.bfloat16)
            with torch.no_grad():
                generated_ids = self.verify_mllm['model'].generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            verdict = self.verify_mllm["processor"].batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        elif self.verify_mllm_name in ['llava','idefics','llava-inter']:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (self.prompt_temp % (meta_info, claim_txt))}
                    ],
                    }
                ]
            num_images=len(claim_img)
            image_list=[]
            for img in claim_img[:num_orig_claim_imgs]:
                image_list.append(Image.open(img).convert('RGB'))
                messages[0]['content'].insert(-1,{"type": "image"})
            evid_texts=[]
            for i,evid in enumerate(evid_context):
                evid_text=evid['text']
                evid_images=evid['images']
                if len(evid_images)==0:
                    #evid_texts.append(str(i+1)+". "+evid_text)
                    messages[0]["content"].append({'type':'text','text':str(i+1)+". "+evid_text})
                else:
                    img_token_list=re.findall(r"\[IMG_.*?\]",evid_text) # [IMG_1], [IMG_2]...
                    if len(img_token_list)==0:
                        messages[0]["content"].append({'type':'text','text':str(i+1)+". "+evid_text})
                    else:
                        split_string=split_string_by_words(evid_text,img_token_list)
                        for m,sp_str in enumerate(split_string):
                            if sp_str in img_token_list:
                                img_idx=re.findall(r'\d+',sp_str)[0]
                                try:
                                    image_list.append(Image.open(evid_images[int(img_idx)-1]).convert('RGB'))
                                    messages[0]["content"].append({'type':'image'})
                                except:
                                    print ('Dealing with edge cases!!! Skip invalid image ids')
                                    continue
                                #    num_images+=1
                            else:
                                if m==0:
                                    messages[0]["content"].append({'type':'text','text':str(i+1)+". "+sp_str})
                                else:
                                    messages[0]["content"].append({'type':'text','text':sp_str})
            messages[0]["content"].append({'type':'text','text':"\nVerdict:"})
            #evid_texts=' '+' '.join(evid_texts)
            #messages[0]['content'][-1]['text']+=evid_texts
            prompt=self.verify_mllm["processor"].apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.verify_mllm["processor"](
                images=image_list,
                text=prompt, padding=True, return_tensors="pt").to(self.verify_mllm['model'].device)
            if self.verify_mllm_name=='llava-inter':
                inputs=inputs.to(torch.float16)
            
            if self.verify_mllm_name=='llava':
                output = self.verify_mllm['model'].generate(**inputs, max_new_tokens=128,
                                                            pad_token_id=self.verify_mllm["processor"].tokenizer.eos_token_id,
                                                            eos_token_id=[
                                                                self.verify_mllm["processor"].tokenizer.eos_token_id,
                                                                self.verify_mllm["processor"].tokenizer.convert_tokens_to_ids("<|eot_id|>")][0])
                verdict=self.verify_mllm["processor"].decode(output[0], 
                                                             skip_special_tokens=True, 
                                                             clean_up_tokenization_spaces=False).split('[/INST]')[-1].strip()
            elif self.verify_mllm_name=='llava-inter':
                output = self.verify_mllm['model'].generate(**inputs, max_new_tokens=128,
                                                            pad_token_id=self.verify_mllm["processor"].tokenizer.eos_token_id,
                                                            eos_token_id=[
                                                                self.verify_mllm["processor"].tokenizer.eos_token_id,
                                                                self.verify_mllm["processor"].tokenizer.convert_tokens_to_ids("<|endoftext|>")])
                verdict=self.verify_mllm["processor"].decode(output[0][len(inputs.input_ids[0]):], 
                                                             skip_special_tokens=True, 
                                                             clean_up_tokenization_spaces=False).strip() 
            elif self.verify_mllm_name=='idefics':
                verdict=self.verify_mllm["processor"].decode(output[0], 
                                                             skip_special_tokens=True, 
                                                             clean_up_tokenization_spaces=False).split('Assistant: ')[-1].strip()
        elif self.verify_mllm_name=='paligemma':
            prompt=(self.prompt_temp % (meta_info, claim_txt))
            image_list=[]
            for img in claim_img[:num_orig_claim_imgs]:
                im=Image.open(img)
                image_list.append(im)
            evid_texts=[]
            for i,evid in enumerate(evid_context):
                evid_text=evid['text']
                evid_images=evid['images']
                evid_texts.append(str(i+1)+". "+evid_text)
                if len(evid_images)==0:
                    continue
                else:
                    img_token_list=re.findall(r"\[IMG_.*?\]",evid_text) # [IMG_1], [IMG_2]...
                    split_string=split_string_by_words(evid_text,img_token_list)
                    """
                    TODO: update max number of images ==> using larger GPU cards
                    """
                    for sp_str in split_string:
                        if sp_str in img_token_list:
                            img_idx=re.findall(r'\d+',sp_str)[0]
                            #if num_images<5:
                            image_list.append(Image.open(evid_images[int(img_idx)-1]).convert('RGB'))
                            #    num_images+=1
                        else:
                            continue
            evid_texts=' '+' '.join(evid_texts)
            prompt+=evid_texts
            prompt+="\nVerdict:"
            inputs = self.verify_mllm["processor"](images=[image_list], text=prompt, return_tensors="pt").to(self.verify_mllm['model'].device)
            output = self.verify_mllm['model'].generate(**inputs, max_new_tokens=128)
            ques=self.verify_mllm["processor"].decode(output[0], skip_special_tokens=True)[inputs.input_ids.shape[1]: ]
        """
        if self.verify_mllm_name in ['llava','qwen']:
            for message in messages[0]['content']:
                print (message['type'])
                if message['type']=='text':
                    print ('\t',message['text'])
                else:
                    print ('\t[IMG] ')
        """
        return verdict