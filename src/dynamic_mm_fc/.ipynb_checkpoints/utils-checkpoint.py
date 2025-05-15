import re

def parse_init_ques(ques):
    ques=ques.strip()
    img_idx=re.findall(r"Image index:\s*([\d,]+)\.", ques)
    flag=False
    if len(img_idx)==0:
        flag=True
        img_idx=re.findall(r"Image index:\s*([\d,]+)", ques)
    if len(img_idx):
        #print ("Image index:"+img_idx[0])
        if flag:
            filter_ques=ques.replace("Image index: "+img_idx[0],'')
        else:
            filter_ques=ques.replace("Image index: "+img_idx[0]+'.','')
        img_idx=re.findall('\d+',img_idx[0])
    else:
        filter_ques=ques
        img_idx=None
    ques_txt=filter_ques.replace('Question: ','')
    if img_idx is None:
        img_idx=['1']
    ques_type="**Image-related:**"
    if img_idx is not None:
        img_idx=[idx.strip() for idx in img_idx]
    return ques_txt.strip(), img_idx,ques_type

def parse_follow_ques(ques):
    ques=ques.strip()
    img_idx=re.findall(r"\*\*Image Index:\*\*\s*([\d,\s]+)\.", ques)
    flag=False
    if len(img_idx)==0:
        img_idx=re.findall(r"\*\*Image Index:\*\*\s*([\d,\s]+)", ques)
        flag=True
    #print (img_idx)
    if len(img_idx):
        #print ("**Image Index:**"+img_idx[0],("**Image Index:**"+img_idx[0]) in ques)
        if flag:
            filter_ques=ques.replace("**Image Index:** "+img_idx[0],'')
        else:
            filter_ques=ques.replace("**Image Index:** "+img_idx[0]+'.','')
        img_idx=re.findall('\d+',img_idx[0])
        #print (filter_ques,'##')
    else:
        filter_ques=ques
        img_idx=None
    ques_type=re.findall(r"\*\*.*?\*\*", filter_ques)
    if len(ques_type)==0:
        ques_type=="**Image-related:**"  
        ques_txt=None
    else:
        ques_txt=""
    ques_type=ques_type[0]
    if ques_type=="**Image-related:**" and img_idx is None:
        img_idx=['1']
    if ques_txt is not None:
        ques_txt=filter_ques.replace(ques_type,'').replace('[QUES] ','').strip()
    if img_idx is not None:
        img_idx=[idx.strip() for idx in img_idx]
    return ques_txt, img_idx, ques_type

def parse_ques(ques,i, para_ques):
    #print (ques)
    if i==0 and para_ques==False:
        ques_txt, img_idx, ques_type=parse_init_ques(ques)
    else:
        ques_txt,img_idx,ques_type=parse_follow_ques(ques)
    ques_txt=ques_txt.split('\n')[0].strip()
    return ques_txt,img_idx,ques_type

def split_string_by_words(text, word_list):
    # Create a regex pattern with word boundaries for each word in the list
    pattern = r'(' + r'|'.join(map(re.escape, word_list)) + r')'
    # Use re.split to split the text and keep the delimiters
    split_result = re.split(pattern, text)
    # Remove empty strings and strip spaces
    split_result = [s.strip() for s in split_result if s.strip()]
    return split_result
