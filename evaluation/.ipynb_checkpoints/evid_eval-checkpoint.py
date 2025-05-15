import json
import os
import pickle as pkl
import random
import argparse

import sys
sys.path.append('..')
root_dir=os.path.abspath('..')
from ref_eval import val_evid_idv, compute_image_scores
from qa_to_evidence import qa_to_evid

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

import re

def convert_qa_format(question_info, llm, llm_name,human):
    answers=question_info["answers"]
    ques_txt=question_info['question'].replace('\n','; ')
    related_images=[]
    ques_img_str=[]
    ans_text=[]
    if (len(question_info['input_images'])):
        rel_images=question_info["input_images"]
        for image in rel_images:
            if human:
                related_images.append(os.path.join(root_dir,'data/combined_images',image[3:]))
            else:
                related_images.append(os.path.join(root_dir,'data/data_clean/images',image))
            ques_img_str.append('[IMG_'+str(len(related_images))+']')
    for j,answer in enumerate(answers):
        answer_type=answer["answer_type"]
        if answer_type=='Image':
            image_answers=answer["image_answers"]
            for image in image_answers:
                if human:
                    related_images.append(os.path.join(root_dir,'data/combined_images',image[3:]))
                else:
                    related_images.append(os.path.join(root_dir,'data/data_clean/images',image))
                ans_text.append('[IMG_'+str(len(related_images))+']')
        else:
            ans_text.append(answer["answer_text"])
        if answer_type=='Boolean':
            boolean_explanation=answer["boolean_explanation"]
            ans_text.append(boolean_explanation) 
    ans_text=' '.join(ans_text).replace('\n','; ')
    if len(ques_img_str):
        evid_ques=ques_txt +', '.join(ques_img_str)
    else:
        evid_ques=ques_txt
    evid=qa_to_evid(evid_ques,ans_text,
                    llm,llm_name)
    #print (evid)
    #print (related_images)
    evid_info={
        'text':evid,
        'images':related_images
    }
    return evid_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate extra questions based on claims with a prompt. Useful for searching.')
    parser.add_argument('--eval_model', 
                        default="gemini")
    parser.add_argument('--llm_name', 
                        default="gemini-2.0-flash-001")
    parser.add_argument('--mllm_name', 
                        default="gemini-2.0-flash-001")
    parser.add_argument('--root_dir', 
                        default="")#this is the absolute path where you put AVerImaTec.
    parser.add_argument('--save_num', 
                        type=str,
                        default="4")
    parser.add_argument('--eval_type', 
                        type=str,
                        default="evidence")
    parser.add_argument('--debug', 
                        type=bool,
                        default=False)
    parser.add_argument('--seperate_val', 
                        type=bool,
                        default=False)
    parser.add_argument('--human_pred', 
                        type=bool,
                        default=False)
    parser.add_argument('--text_val', 
                        type=bool,
                        default=False)#evaluating only on the textual part of evidence
    args = parser.parse_args()

    from google import genai
    from private_info import API_keys
    from google.genai.types import HttpOptions
    mllm = genai.Client(http_options=HttpOptions(api_version="v1"),api_key=API_keys.GEMINI_API_KEY)
    mllm_name='gemini-2.0-flash-001'

    p2_data=load_json(os.path.join(args.root_dir,'data/data_clean/split_data/test.json'))
    if args.human_pred==False:
        print ('Loading model predictions...')
        save_str='_'.join([args.llm_name,args.mllm_name])
        pred_file=load_pkl(os.path.join(args.root_dir,'fc_detailed_results','_'.join([args.llm_name,args.mllm_name]),str(args.save_num)+'.pkl'))
        if os.path.exists(os.path.join(args.root_dir,
                                            "evaluation",
                                            'intermediate_info/'+save_str+'_val_evid_'+str(args.save_num)+'_raw.pkl')):
            results=load_pkl(os.path.join(args.root_dir,
                                            "evaluation",
                                            'intermediate_info/'+save_str+'_val_evid_'+str(args.save_num)+'_raw.pkl'))
        else:
            results={}
    else:
        print ('Loading evidence annotated by two annotators...')
        save_str='human_anno'
        #better arrange everything in a folder
        pred_file=load_pkl(os.path.join(root_dir,'data/combined_images','second_inter_results.pkl'))
        print ('Length of evaluation:',len(pred_file))
        results={}

    test_id_to_row={}
    for i,row in enumerate(p2_data):
        test_id_to_row[i]=row

    
    vis=0
    for req_id in pred_file:
        if req_id in results:
            print('Already there... Skip!')
            continue
        if args.debug:
            if vis>5:
                break
        vis+=1
        if vis%20==0:
            pkl.dump(results,open(os.path.join(args.root_dir,
                                               "evaluation",
                                               'intermediate_info/'+save_str+'_val_evid_'+str(args.save_num)+'_raw.pkl'),'wb'))
        if args.human_pred:
            pred_info=pred_file[req_id]['first_round']['questions']
            ref_info=pred_file[req_id]['second_round']['questions']
            #print (pred_file[req_id]['first_round'])
            #print (ref_info)
            pred_evid=[convert_qa_format(qa,mllm,mllm_name,args.human_pred) for qa in pred_info]
        else:
            pred_info=pred_file[req_id]
            ref_info=test_id_to_row[req_id]['questions']
            if len(ref_info)==0:
                continue
            pred_evid=pred_info['evidence'] #[{'images':xxx,'text':xxx}]
        #converting QA to evidence ==> for model predictions, no need
        ref_evid=[convert_qa_format(qa,mllm,mllm_name,args.human_pred) for qa in ref_info]
        print (len(ref_evid),len(pred_evid))
        detailed_val, val_score=val_evid_idv(mllm, pred_evid, ref_evid, args.text_val, args.seperate_val)

        #print (detailed_val,val_score)
        results[req_id]={
            'ref_evid':ref_evid,
            'pred_evid':pred_evid,
            'detailed_val':detailed_val,
            'val_score':val_score
        }
        if args.seperate_val:
            img_scores=compute_image_scores(mllm,pred_evid,ref_evid,val_score)

            results[req_id]['image_scores']=img_scores
        #print (results[req_id])
    pkl.dump(results,open(os.path.join(args.root_dir,
                                            "evaluation",
                                            'intermediate_info/'+save_str+'_val_evid_'+str(args.save_num)+'_raw.pkl'),'wb'))
