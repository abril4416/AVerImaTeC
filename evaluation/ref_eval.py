import os
import pickle as pkl
import torch
from PIL import Image
import sys
sys.path.append('..')
import re
import random

angles=[Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270]
resizes=[(300,300),(100,500),(500,100)]

root_dir=os.path.abspath('..')
text_val_demo=open(os.path.join(root_dir,
                                "templates/evid_evaluation_text.txt")).readlines()
text_val_demo="".join(text_val_demo)

seperate_val_demo=open(os.path.join(root_dir,
                                "templates/evid_evaluation_text_seperate.txt")).readlines()
seperate_val_demo="".join(seperate_val_demo)

from templates import evid_evaluation_joint
joint_val_demo=''.join([evid_evaluation_joint.instruction,
                        evid_evaluation_joint.exp_first,
                        evid_evaluation_joint.exp_second,])

def split_string_by_words(text, word_list):
    # Create a regex pattern with word boundaries for each word in the list
    pattern = r'(' + r'|'.join(map(re.escape, word_list)) + r')'
    # Use re.split to split the text and keep the delimiters
    split_result = re.split(pattern, text)
    # Remove empty strings and strip spaces
    split_result = [s.strip() for s in split_result if s.strip()]
    return split_result

def gen_incontext_input_textonly(pred,ref,demos):
    texts=[]
    texts.append(demos)
    texts.append("\n[PRED]: "+pred)
    texts.append("[REF]: "+ref)
    texts='\n'.join(texts)
    return texts

def score_extraction(feedback):
    pred_in_ref=feedback.split('[PRED in REF]: ')[-1].split('\n')[0].split(';')[0].strip()
    ref_in_pred=feedback.split('[REF in PRED]: ')[-1].split('\n')[0].split(';')[0].strip()
    if pred_in_ref.isdigit():
        pred_in_ref=int(pred_in_ref)
    else:
        pred_in_ref=0
    if ref_in_pred.isdigit():
        ref_in_pred=int(ref_in_pred)
    else:
        ref_in_pred=0
    score={
        'ref_in_pred':ref_in_pred,
        'pred_in_ref':pred_in_ref
    }
    if len(feedback.split('[PRED in REF]: ')[-1].split('\n')[0].split(';')):
        score['detailed_ref_in_pred']=';'.join(feedback.split('[REF in PRED]: ')[-1].split('\n')[0].split(';')[1:]).strip()
        score['detailed_pred_in_ref']=';'.join(feedback.split('[PRED in REF]: ')[-1].split('\n')[0].split(';')[1:]).strip()
    return score

def seperate_text_val(gt_set, pred_set,
                      path, llm_name, mllm_name, save_num, 
                      debug_mode=False, eval_type=None):
    #scores={user:{} for user in all_users_pred}
    demonstrations=open(os.path.join(path,"templates/ques_evaluation_text.txt")).readlines()
    demonstrations="".join(demonstrations)
    
    from google import genai
    from private_info import API_keys
    from google.genai.types import HttpOptions
    model = genai.Client(http_options=HttpOptions(api_version="v1"),api_key=API_keys.GEMINI_API_KEY)
    
    raw_response={}
    processed_response={}
    for req_id in pred_set:
        if llm_name=='gemini-2.5-pro-exp-03-25':
            pred=[row for k,row in enumerate(pred_set[req_id])]
        else:
            pred=[str(k+1)+'. '+row for k,row in enumerate(pred_set[req_id])]
        gt=[str(k+1)+'. '+row for k,row in enumerate(gt_set[req_id])]
        
        ref=' '.join(gt)
        pred=' '.join(pred)
        
        print ("###",req_id,"###")
        print ('GT evid:\n\t',ref)
        print ('Pred evid:\n\t',pred)
        incontext_input=gen_incontext_input_textonly(pred,ref,demonstrations)
        response = model.models.generate_content(
            #model='gemini-2.5-pro-exp-03-25',
            model='gemini-2.0-flash-001',
            contents=incontext_input
        )
        feedback=response.text
        
        processed_score=score_extraction(feedback)
        raw_response[req_id]=feedback
        processed_response[req_id]=processed_score
        if debug_mode:
            print (feedback,'\n\n')
    pkl.dump(raw_response,open(os.path.join(path,
                               "evaluation",
                               'intermediate_info/'+'_'.join([llm_name,mllm_name])+'_val_text_'+str(save_num)+'_raw.pkl'),'wb'))
    pkl.dump(processed_response,open(os.path.join(path,
                                     "evaluation",
                                     'intermediate_info/'+'_'.join([llm_name,mllm_name])+'_val_text_'+str(save_num)+'_processed.pkl'),'wb'))
    return 

def gen_img_text_split(evid_context,pred=False):
    inputs=[]
    for i,evid in enumerate(evid_context):
        evid_text=evid['text']+' '
        if i==0 and pred:
            evid_text="[REF]: "+evid_text
        elif i==0:
            evid_text='[PRED]: '+evid_text
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
                        """
                        manip_type=random.randint(0,1)
                        if manip_type==0:
                            rotate_angle=random.randint(0,2)
                            inputs.append(Image.open(evid_images[int(img_idx)-1]).convert('RGB').transpose(angles[rotate_angle]))
                        else:
                           resize_type=random.randint(0,2)
                           inputs.append(Image.open(evid_images[int(img_idx)-1]).convert('RGB').resize(resizes[resize_type]))
                        """
                        inputs.append(Image.open(evid_images[int(img_idx)-1]).convert('RGB'))
                    else:
                        if m==0:
                            inputs.append(str(i+1)+". "+sp_str)
                        else:
                            inputs.append(sp_str)
    return inputs
    
def val_evid_idv(model, pred_evid, ref_evid, text_val, seperate_val):
    pred=[str(k+1)+'. '+row['text'] for k,row in enumerate(pred_evid)]
    gt=[str(k+1)+'. '+row['text'] for k,row in enumerate(ref_evid)]
    ref='. '.join(gt)
    pred='. '.join(pred)
    if text_val or seperate_val:
        #print ('GT evid:\n\t',ref)
        #print ('Pred evid:\n\t',pred)
        if seperate_val:
            #print ('Seperation!')
            incontext_input=gen_incontext_input_textonly(pred,ref,seperate_val_demo)
        else:
            incontext_input=gen_incontext_input_textonly(pred,ref,text_val_demo)
        response = model.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=incontext_input
            )
        feedback=response.text
    else:
        #consider inter-leaved image-text evaluation
        inputs=[joint_val_demo]
        ref_split=gen_img_text_split(ref_evid)
        pred_split=gen_img_text_split(pred_evid,pred=True)
        inputs.extend(ref_split)
        inputs.extend(pred_split)
        response = model.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=inputs
            )
        feedback=response.text
    processed_score=score_extraction(feedback)
    return feedback, processed_score

def compute_image_scores(model,pred_evid,ref_evid,score):
    prompt="Given two sets of images, you need to score how similar they are, ranging from 0-10. The number of images could be different in image sets.\n"
    prompt+='[IMG_SET_1]:'
    ref_in_pred=re.findall(r'\(.*?\)',score['detailed_ref_in_pred'])
    pred_in_ref=re.findall(r'\(.*?\)',score['detailed_pred_in_ref'])
    print (ref_in_pred,pred_in_ref)
    image_scores={'pred_in_ref':[],'ref_in_pred':[]}
    for detail in pred_in_ref:
        info=detail[1:-1].split(',')
        try:
           pred_idx=int(info[0].split('_')[-1])
           ref_idx=int(info[1].split('_')[-1])
           imgs_pred=pred_evid[pred_idx-1]['images']
           imgs_ref=ref_evid[ref_idx-1]['images']
           if len(imgs_pred)==0 or len(imgs_ref)==0:
               feedback='10'
           else:
               inputs=[prompt]
               for img in imgs_pred:
                   """
                   manip_type=random.randint(0,1)
                   if manip_type==0:
                       rotate_angle=random.randint(0,2)
                       inputs.append(Image.open(img).convert('RGB').transpose(angles[rotate_angle]))
                   else:
                       resize_type=random.randint(0,2)
                       inputs.append(Image.open(img).convert('RGB').resize(resizes[resize_type]))
                   """
                   inputs.append(Image.open(img).convert('RGB'))
               inputs.append('\n[IMG_SET_2]:')
               for img in imgs_ref:
                   inputs.append(Image.open(img).convert('RGB'))
               inputs.append('\nPlease generate your rating with one integer:')
               response = model.models.generate_content(
                   model='gemini-2.0-flash-001',
                   contents=inputs
                  )
               feedback=response.text
        except:
            print ('##Edge case image!!')
            feedback='10'
        image_scores['pred_in_ref'].append({
            'info':info,
            'score':feedback
        })
    for detail in ref_in_pred:
        info=detail[1:-1].split(',')
        try:
           pred_idx=int(info[0].split('_')[-1])
           ref_idx=int(info[1].split('_')[-1])
           imgs_pred=pred_evid[pred_idx-1]['images']
           imgs_ref=ref_evid[ref_idx-1]['images']
           if len(imgs_pred)==0 or len(imgs_ref)==0:
               feedback='10'
           else:
               inputs=[prompt]
               for img in imgs_pred:
                   inputs.append(Image.open(img).convert('RGB'))
               inputs.append('\n[IMG_SET_2]:')
               for img in imgs_ref:
                  inputs.append(Image.open(img).convert('RGB'))
               inputs.append('\nPlease generate your rating with one integer:')
               response = model.models.generate_content(
                   model='gemini-2.0-flash-001',
                   contents=inputs
                   )
               feedback=response.text
        except:
            print ('##Edge case image!!')
            feedback='10'
        image_scores['ref_in_pred'].append({
            'info':info,
            'score':feedback
        })
    return image_scores