from dynamic_mm_fc.utils import parse_ques
import dynamic_mm_fc.tools as tools
import re

"""
In order to reduce computation load
    using a single MLLM and a single LLM
        for different purposes
"""
from dynamic_mm_fc.conv_utils import qa_to_evidence

class QA_Model:

    def __init__(self, planner, answer_llm, answer_mllm, answer_llm_name, answer_mllm_name, save_num, debug, path, no_search):
        self.planner= planner #select answering method: reverse image search; texutal search; # image comparison

        self.answer_llm=answer_llm
        self.answer_mllm=answer_mllm
        self.answer_llm_name=answer_llm_name
        self.answer_mllm_name=answer_mllm_name

        self.save_num=save_num
        if save_num in [4,5]:
            self.para_ques=True
        else:
            self.para_ques=False
        self.debug=debug
        self.path=path #root_path
        self.no_search=no_search

        self.tool_mapper={
            "A": tools.reverse_image_search,
            "B": tools.vqa_mllm,
            "C": tools.text_search_text,
            "D": tools.text_search_image
        }

        self.method_mapper={
            "A": "Reverse Image Search",
            "B": "Visual Question Answering",
            "C": "Text Search for Texts",
            "D": "Text Search for Images"
        }

    def tool_executor(self, claim_txt, claim_img,
                      ques_txt, ques_type, img_idx,
                      ans_method,
                      meta_date, meta_loc):
        evid_url=None #initialization; for VQA, no evid urls
        if ans_method in ["B","D"]:
            answer, detailed_evid, evid_url=self.tool_mapper[ans_method](ques_txt, img_idx, 
                                                                         claim_txt, claim_img,
                                                                         meta_date, meta_loc,
                                                                         self.answer_mllm,self.answer_mllm_name,self.path)
        elif ans_method in ["A","C"]:
            answer, detailed_evid, evid_url=self.tool_mapper[ans_method](ques_txt, img_idx, 
                                                                         claim_txt, claim_img,
                                                                         meta_date, meta_loc,
                                                                         self.answer_llm,self.answer_llm_name,self.path)
        return answer, detailed_evid, evid_url

    def answer_ques(self, claim_txt, claim_img,
                    evid_context, ques_txt, img_idx, ques_type,
                    meta_date, meta_loc,tool_icl=False, prev_ques_info=None):
        
        if img_idx is not None and len(img_idx)>3:
            img_idx=img_idx[:3] #deail with hallucinations of models
        if self.debug:
            print ('\t###Detailed info:',ques_txt, ques_type, img_idx)
        
        if img_idx is None or len(img_idx)==0:
            ques_input=ques_txt
            related_images=[]
        else:
            #all_ids_raw=re.findall(r'\d+',img_idx)
            all_ids=[]
            for idx in img_idx:
                if int(idx)<=len(claim_img):
                    all_ids.append(idx)
                else:
                    all_ids.append('1')#exception error case
            ques_input=ques_txt+' Image index: '+','.join(img_idx)
            related_images=[claim_img[int(idx)-1] for idx in all_ids]

        """
        if prev_ques_info is not None:
            #no search baseline
            ans_method=prev_ques_info['answer_method']
            if ans_method in ['A','C','D']:
                answer='No answer could be found.'
            else:
                answer=prev_ques_info['answer']
            detailed_evid=None
            evid_url=None
            if img_idx is not None and len(img_idx)>0:
                evid_ques=ques_txt+", ".join(["[IMG_"+str(i+1)+']' for (i,_)  in enumerate(all_ids)])
            else:
                evid_ques=ques_txt
        """    
        try:
            if prev_ques_info is not None:
                #no search baseline
                ans_method=prev_ques_info['answer_method']
            else:
                ans_method= self.planner.gen_next_plan(ques_input, ques_type, tool_icl)
        except:
            #detail with exceptions: %d/%o in demonstrations
            if ques_type=='Image-related':
                ans_method='A'
            else:
                ans_method='C'
        if ans_method not in ['A','B','C','D']:
            if ques_type=='Image-related':
                ans_method='A'
            else:
                ans_method='C'
        if ans_method in ['A','B'] and img_idx is None:#deal with edge case
            print ('###Edge case!!!! Tools in need of input images while the question is not image related!')
            img_idx=['1']
            all_ids=[]
            for idx in img_idx:
                if int(idx)<=len(claim_img):
                    all_ids.append(idx)
                else:
                    all_ids.append('1')#exception error case
            ques_input=ques_txt+' Image index: '+','.join(img_idx)
            related_images=[claim_img[int(idx)-1] for idx in all_ids]
            ques_type="**Image-related:**"
        if self.debug:
            print ("###Answering Stage###")
            print ('\tQues info:',ques_txt, 'Related image:', img_idx)
            print ('\tQues type:',ques_type)
            print ('\tAnswer tool:',self.method_mapper[ans_method])
        if prev_ques_info is not None:
            if ans_method in ['A','C']:
                answer='No answer could be found.'
            elif ans_method=='D':
                answer=[]
            else:
                answer=prev_ques_info['answer']
            detailed_evid=None
            evid_rul=None
        else:
            answer, detailed_evid, evid_url=self.tool_executor(claim_txt, claim_img,
                                                               ques_txt, ques_type, img_idx,
                                                               ans_method,
                                                               meta_date, meta_loc)
        if ans_method=="D" and len(answer)>0: # augment the claim image set!
            claim_img.append(answer)
            evid_ans="[IMG_"+str(len(related_images)+1)+']'
            related_images.append(answer)
        elif ans_method=="D" and len(answer)==0: 
            evid_ans='No answer could be found.'
        else:
            evid_ans=answer
        if img_idx is not None and len(img_idx)>0:
            evid_ques=ques_txt+", ".join(["[IMG_"+str(i+1)+']' for (i,_)  in enumerate(all_ids)])
        else:
            evid_ques=ques_txt
        """
        this needs to be updated!!!
            Image path incorporated ==> also verifier, justification generation
        """
        evid=qa_to_evidence.qa_to_evid(evid_ques,evid_ans,
                                       self.answer_llm,self.answer_llm_name,self.path)
        print ('Statement:',evid)
        evid_context.append({
            'text':evid,
            'images':related_images
        })
        
        if self.debug:
            
            print ('\tAnswer:',answer)
            print ('\tCur Evidence:','; '.join([evid['text'] for evid in evid_context]))
        return answer, evid, ans_method