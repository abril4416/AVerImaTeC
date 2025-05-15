import re

def parse_init_ques(ques):
    info=ques.split('\n')
    ques_txt=info[0].replace('Question: ','').strip()
    img_idx=info[1].split(': ')[-1].replace('.','').strip()
    return ques_txt, img_idx

def parse_follow_ques(ques):
    info=re.findall(r"\*\*.*?\*\*", ques)
    #if len(info)==1: text-related or image-related
    if len(info)==1:
        img_idx=None
    else:
        text_info=ques.replace(info[0],'')
        ques_txt=text_info.split(info[1])[0].strip()
        img_idx=text_info.split(info[1])[-1].strip()
    ques_type=info[0].replace('*','').replace(':','').strip()
    return ques_txt, ques_type, img_idx