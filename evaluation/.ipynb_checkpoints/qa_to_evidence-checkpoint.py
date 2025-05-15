import os
root_dir=os.path.abspath('..')
import torch

def gen_incontext_input(ques,ans,demos):
    texts=[]
    texts.append(demos)
    texts.append("[QUES]: "+ques)
    texts.append("[ANS]: "+ans)
    texts.append("[STAT]:")
    texts='\n'.join(texts)
    return texts

def qa_to_evid(ques, ans, llm,llm_name):
    
    #loading demonstrations
    demonstrations=open(os.path.join(root_dir,"templates/qa_to_evid_demos.txt")).readlines()
    demonstrations="\n".join(demonstrations)
    incontext_input=gen_incontext_input(ques,ans,demonstrations)
    if "gemini" in llm_name:
        response = llm.models.generate_content(
            model=llm_name,
            contents=incontext_input
        )
        statement=response.text
        statement=statement.replace('[STAT]:','').strip()
    return statement