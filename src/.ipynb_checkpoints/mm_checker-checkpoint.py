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

class MM_Checker:

    def __init__(self,qg_model,qa_model,verifier,justification_gen, 
                 max_qa_iter,num_gen_ques,max_invalid,max_num_images,
                 qg_icl,tool_icl,mllm_name,
                 gt_ques,gt_evid,para_qg,hybrid_qg,no_search,
                 debug,path):
        #expert initialization
        self.qg_model= qg_model #generate questions based on the history evidence
        self.qa_model= qa_model #answer questions (indipendently)
        self.verifier= verifier #verify if a verdict can be reached
        self.justification_gen= justification_gen #generating a justification based on the evidence and claim

        self.max_qa_iter=max_qa_iter
        self.num_gen_ques=num_gen_ques
        self.max_invalid=max_invalid
        self.max_num_images=max_num_images

        #whether leveraging few-shot for 1) question generation and 2) tool selection
        self.qg_icl=qg_icl
        self.tool_icl=tool_icl
        self.mllm_name=mllm_name
        
        self.gt_ques=gt_ques
        self.gt_evid=gt_evid
        self.para_qg=para_qg
        self.hybrid_qg=hybrid_qg
        self.no_search=no_search
        
        self.debug=debug
        self.path=path

    def convert_gt_to_question_temp(gt_info, claim_txt, claim_img):
        questions=[]
        return questions

    def verify_mm_claim(self, claim_txt, claim_img, 
                        meta_date, meta_loc, gt_info=None):
        #be careful: the claim_img here ais path (for compatibility)
        evid_context=[]
        justification=None

        meta_info=META_EVID % (meta_date, meta_loc)
        max_iter=self.max_qa_iter
        if len(claim_img)>self.max_num_images:
            claim_img=claim_img[:self.max_num_images]
        num_orig_claim_imgs=len(claim_img)#seperate claim images to candidate images
        if self.gt_ques:
            questions=self.convert_gt_to_question_temp(gt_info, claim_txt, claim_img)
            max_iter=len(questions)
        elif self.no_search:
            questions=gt_info
            max_iter=len(questions)
        elif self.para_qg or self.hybrid_qg:
            questions=[]
            candidate_questions=self.qg_model.generate_parallel_ques(claim_txt, claim_img, self.num_gen_ques, self.qg_icl)
            for k,ques in enumerate(candidate_questions):
                try:
                    ques_txt,img_idx,ques_type=parse_ques(ques,k, self.para_qg)
                    if ques_txt is None or len(ques_txt.split(' '))<5:
                        continue
                    questions.append(ques)
                    if self.debug:
                        print (k+1,ques)
                except:
                    print ('###Invalid in the pre-filtering stage:',ques)
                    continue
            if max_iter>len(questions) and self.hybrid_qg==False:
                max_iter=len(questions)
        num_invalid=0
        detailed_results={
            'QA_info':[],
            'evidence':None
        }

        for iter in range(max_iter):
            prev_ques_info=None
            cur_results={
                'raw_questions':None,
                'question':None,
                'answer':None,
                'ques_type':None,
                'answer_method':None,
            }
            if self.debug:
                print ('Iteration:',iter)

            #both are for raw questions
            if self.para_qg or self.no_search or self.gt_ques or (self.hybrid_qg and iter<len(questions)):
                if self.debug and self.hybrid_qg:
                    print (iter,'Parallel##')
                if self.no_search:
                    ques=questions[iter]['raw_questions']
                    prev_ques_info=questions[iter]
                else:
                    ques=questions[iter]
            else:
                if  self.debug and self.hybrid_qg:
                    print (iter,'Dynamic##')
                ques=self.qg_model.gen_question(evid_context,
                                                claim_txt, claim_img, 
                                                meta_date, meta_loc,num_orig_claim_imgs, self.qg_icl)
            if self.debug:
                print (iter+1,'Raw question:',ques)
            #edge cases: empirically found
            if ques is None or len(ques)<10:
                continue
            try:
                ques_txt,img_idx,ques_type=parse_ques(ques,len(detailed_results['QA_info']), self.para_qg)
                if ques_txt is None or len(ques_txt.split(' '))<5:
                    continue
                #decide question type based on img_idx ==> resolve conflicts
                if img_idx is not None and len(img_idx):
                    ques_type="Image-related"
                else:
                    ques_type="Text-related"
            except:
                print ('Invalid question!!!',ques)
                continue
            ans, evid, ans_method = self.qa_model.answer_ques(claim_txt, claim_img,
                                                              evid_context, ques_txt, img_idx, ques_type,
                                                              meta_date, meta_loc, self.tool_icl, prev_ques_info)
            if ans is None:
                print ('Invalid question!! Edge case')
                continue
            cur_results['raw_questions']=ques
            cur_results['answer']=ans
            cur_results['answer_method']=ans_method
            cur_results['ques_type']=ques_type
            cur_results['question']=ques_txt
            detailed_results['QA_info'].append(cur_results)
            if 'llava' in self.mllm_name:
                torch.cuda.empty_cache()

        cur_verdict=self.verifier.verify(meta_info, claim_txt, claim_img, evid_context,num_orig_claim_imgs)
        justification=self.justification_gen.justi_gen(cur_verdict,meta_info,claim_txt, claim_img, evid_context,num_orig_claim_imgs)
        detailed_results['evidence']=evid_context
        detailed_results['verdict']=cur_verdict
        detailed_results['justification']=justification
        if self.debug:
            print ('Verdict:',cur_verdict)
            print ('Justification:',justification)
        return detailed_results

def predict_verdict_with_gt_evid(verifier, justification_gen,
                                 gt_info, claim_txt, claim_img,
                                 meta_date, meta_loc,
                                 answer_llm, answer_llm_name, path):
    if len(claim_img)>3:
        claim_img=claim_img[:3]
    num_orig_claim_imgs=len(claim_img)
    img_dir=os.path.join(os.path.join(path,'data/data_clean/images'))
    candidate_images=[os.path.join(img_dir,img.split('/')[-1]) for img in claim_img]
    meta_info=META_EVID % (meta_date, meta_loc)
    evid_context=[]
    num_imgs=0
    for info in gt_info:
        question=info['question'].strip()
        related_images=[os.path.join(img_dir,img.split('/')[-1]) for img in info['input_images']]
        num_imgs+=len(related_images)
        if len(related_images)==0:
            evid_ques=question
        else:
            evid_ques=question+", ".join(["[IMG_"+str(i+1)+']' for (i,_)  in enumerate(related_images)])
        answers=info['answers']

        evid_ans=[]
        for ans in answers:
            ans_type=ans['answer_type']
            if ans_type=='Image':
                evid_ans.append("[IMG_"+str(len(related_images)+1)+']')
                related_images.append(os.path.join(img_dir,ans['image_answers'][0].split('/')[-1]))
            elif ans_type=='Boolean':
                evid_ans.append(ans['answer_text']+' '+ans['boolean_explanation'].strip())
            else:
                evid_ans.append(ans['answer_text'].strip())
        evid_ans='. '.join(evid_ans)
        print ('Evid Ques:',evid_ques)
        print ('\tEvid Ans:',evid_ans)
        evid=qa_to_evidence.qa_to_evid(evid_ques,evid_ans,
                                       answer_llm,answer_llm_name,path)
        print ('\tconverted evid:',evid)
        evid_context.append({
            'text':evid,
            'images':related_images
        })
    #meta_info, claim_txt, claim_img, evid_context,num_orig_claim_imgs
    verdict=verifier.verify(meta_info, claim_txt, claim_img, evid_context,num_orig_claim_imgs)
    print ('###Verdict:',verdict,'\n')
    justification=justification_gen.justi_gen(verdict,meta_info,claim_txt, claim_img, evid_context,num_orig_claim_imgs)
    return verdict, justification, evid_context

if __name__ == '__main__':
    args=config.parse_opt()

    img_dir=os.path.join(os.path.join(args.ROOT_PATH,'data/data_clean/images'))
    p2_data=load_json(os.path.join(args.ROOT_PATH,'data/data_clean/split_data/test.json'))
    demo_data=load_json(os.path.join(args.ROOT_PATH,'data/data_clean/split_data/train.json'))
    print ('Training data: %d; Testing data:%d' % (len(demo_data),len(p2_data)))
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
    if 'gemini' in llm_name and 'gemini' in mllm_name:
        mllm_model=llm_model
    elif mllm_name=='gemma' and llm_name=='gemma':
        mllm_model=llm_model
    else:
        #https://huggingface.co/blog/smolvlm#performance
        if 'gemini' in mllm_name:
            #import google.generativeai as genai
            import sys
            sys.append('..')
            from private_info.API_keys import GEMINI_API_KEY
            from google import genai
            from google.genai.types import HttpOptions
            mllm_model = genai.Client(http_options=HttpOptions(api_version="v1"), api_key=GEMINI_API_KEY)
        elif mllm_name=='qwen':
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            from qwen_vl_utils import process_vision_info
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", 
                torch_dtype= torch.bfloat16, 
                #device_map="auto"
                device_map={"":"cuda:1"}
            )
            model.train(False)
            min_pixels = 64 * 28 * 28
            max_pixels = 64 * 28 * 28
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                                      min_pixels=min_pixels, max_pixels=max_pixels)
            mllm_model={
                'model':model,
                'processor':processor
            }
        elif mllm_name=='llava':
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            model = LlavaNextForConditionalGeneration.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf', 
                                                                      torch_dtype=torch.float16,
                                                                      device_map={"":"cuda:1"}).eval()
            processor =  LlavaNextProcessor.from_pretrained('llava-hf/llava-v1.6-mistral-7b-hf')
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            mllm_model={
                'model':model,
                'processor':processor
            }
        elif mllm_name=='paligemma':
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
            model_id = "google/paligemma2-10b-mix-448"
            model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, 
                                                                      torch_dtype=torch.bfloat16,
                                                                      device_map={"":"cuda:1"})
            processor = AutoProcessor.from_pretrained(model_id)
            mllm_model={
                'model':model,
                'processor':processor
            }
        elif mllm_name=='idefics':
            from transformers import AutoProcessor, AutoModelForVision2Seq
            processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
            model = AutoModelForVision2Seq.from_pretrained(
                "HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16,
                device_map={"":"cuda:1"}
            )
            mllm_model={
                'model':model,
                'processor':processor
            }
        elif mllm_name=='llava-inter':
            #LLaVA for interleaved image-text comprehension
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            model_id = "llava-hf/llava-interleave-qwen-7b-hf"
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                device_map={"":"cuda:1"}
                )
            processor = AutoProcessor.from_pretrained(model_id)
            mllm_model={
                'model':model,
                'processor':processor
            }
        elif mllm_name=='gemma':
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
            ckpt = "google/gemma-3-12b-it"
            model = Gemma3ForConditionalGeneration.from_pretrained(
                ckpt, device_map="auto", torch_dtype=torch.bfloat16,
            )
            #model=model.bfloat16()
            #processor.padding_side="left"
            processor = AutoProcessor.from_pretrained(ckpt)
            #processor.tokenizer.pad_token = "[PAD]"
            #processor.tokenizer.padding_side = "left"
            mllm_model={
                'model':model.eval(),
                'processor':processor
            }
            

    if args.QG_ICL:
        demo_data_qg=demo_data
    else:
        demo_data_qg=None
    if args.TOOL_ICL:
        demo_data_tool=demo_data
    else:
        demo_data_tool=None
        
    qg_model=qg_model.QG_Model(mllm_model,mllm_name,
                               llm_model, llm_name, args.ROOT_PATH,
                               args.NUM_DEMOS,
                               args.ICL_FEAT,demo_data_qg)
    planner=planner.Plan_Model(llm_model,llm_name,args.DEBUG,args.ICL_FEAT,demo_data_tool)
    qa_model=qa_model.QA_Model(planner, llm_model, mllm_model, llm_name, mllm_name, args.SAVE_NUM, args.DEBUG, args.ROOT_PATH, args.NO_SEARCH)
    verifier=verifier.Verify_Model(mllm_model, mllm_name)
    justification_gen=justification_gen.Justification_Model(mllm_model, mllm_name)

    checker=MM_Checker(qg_model,qa_model,verifier,justification_gen,
                       args.MAX_QA_ITER,args.NUM_GEN_QUES,args.MAX_INVALID,args.MAX_NUM_IMAGES,
                       args.QG_ICL,args.TOOL_ICL,mllm_name,
                       args.GT_QUES,args.GT_EVID,args.PARA_QG,args.HYBRID_QG,args.NO_SEARCH,
                       args.DEBUG,args.ROOT_PATH)

    if os.path.exists(os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name])))==False:
        os.mkdir(os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name])))
    if os.path.exists(os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name]),str(args.SAVE_NUM)+'.pkl')):
        all_results=load_pkl(os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name]),str(args.SAVE_NUM)+'.pkl'))
    else:
        all_results={}

    #all_results={}
    if args.DEBUG:
        #random.shuffle(p2_data)
        p2_data=p2_data[:2]

    counts=0
    if args.NO_SEARCH:
        #paralleled gen questions ==> generally has better generated questions (14 is the save_num)
        prev_pred_file=load_pkl(os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name]),'14.pkl'))
    for i, annotation in enumerate(p2_data):
        req_id=i
        if req_id in all_results:
            continue
        if counts%10==0:
            pkl.dump(all_results,open(os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name]),str(args.SAVE_NUM)+'.pkl'),'wb'))
        counts+=1
        fact_checking_article=annotation["article"]
        if args.GT_EVID:
            questions=annotation["questions"]
        elif args.NO_SEARCH:
            if req_id not in prev_pred_file:
                continue
            questions=prev_pred_file[req_id]['QA_info']
        else:
            questions=None
        claim_txt=annotation["claim_text"].strip()
        claim_images=annotation["claim_images"]
        claim_img=[os.path.join(img_dir,img.split('/')[-1]) for img in claim_images]
        date=annotation['date']#date here is string type
        location=annotation['location']
        if len(location)>0:
            location=pycountry.countries.get(alpha_2=location).name
        else:
            location=""
        print (i,"##Claim##",claim_txt,claim_img,date,location)

        if args.GT_EVID:
            try:
                verdict, justification, evid_context= predict_verdict_with_gt_evid(verifier, justification_gen,
                                                                                   questions, claim_txt, claim_img,
                                                                                   date, location,
                                                                                   llm_model,args.LLM_NAME,args. ROOT_PATH)
            except:
                continue
            detailed_info={
                'QA_info':[],#for gt evid, no need to include it
                'justification':justification,
                'verdict':verdict,
                'evid_context':evid_context,
            }
        else:
            if args.GT_QUES or  args.NO_SEARCH:
                gt_info=questions
            else:
                gt_info=None

            try:
                detailed_info= checker.verify_mm_claim(claim_txt, claim_img, 
                                                           date, location, gt_info)
            except:
                print ('Exception!!!!')
                continue
        
        all_results[req_id]=detailed_info
    pkl.dump(all_results,open(os.path.join(args.ROOT_PATH,'fc_detailed_results','_'.join([llm_name,mllm_name]),str(args.SAVE_NUM)+'.pkl'),'wb'))