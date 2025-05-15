from PIL import Image
import json
import os
import pickle as pkl
import random
from collections import defaultdict
import argparse

import socket
import io
from urllib.parse import urlparse
from htmldate import find_date
from geolite2 import geolite2 
import datetime

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

def get_domain_name(url):
    if '://' not in url:
        url = 'http://' + url

    domain = urlparse(url).netloc

    if domain.startswith("www."):
        return domain[4:]
    else:
        return domain
    
def get_country(url):
    domain_str=get_domain_name(url)
    ip = socket.gethostbyname(domain_str.strip())
    reader = geolite2.reader()      
    output = reader.get(ip)
    result = output['country']['iso_code']
    #origin(ip, domain_str, result)
    return result

def detect_web(client,path,how_many_queries):
    """Detects web annotations given an image."""
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    #content=base64.b64decode(content.encode())
    image = vision.Image(content=content)
    response = client.web_detection(image=image, max_results=how_many_queries)
    return response.web_detection

"""
Be careful about the ref_date format
"""
def compare_date(conv_ref_date,url_date):
    #print (conv_ref_date)
    if conv_ref_date is None:
        conv_ref_date=datetime.date.today()
    conv_ref_date=conv_ref_date.strftime('%Y-%m-%d')
    return url_date<=conv_ref_date

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate extra questions based on claims with a prompt. Useful for searching.')
    parser.add_argument('--file_path', 
                        default="../data/p2_annotation-1-Jan", 
                        )
    parser.add_argument('--num_queries', 
                        type=int,
                        default=100, 
                        )
    parser.add_argument('--users', 
                        default="appen_001,appen_007,appen_009", 
                        )
    args = parser.parse_args()

    p2_results=load_pkl(os.path.join(args.file_path,'processed_result/p2_assignment_annotations_latest.pkl'))
    """
    Get all RIS results for claim images
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    imgs_for_detection=defaultdict(int)
    user_names=args.users.split(',')
    img_to_date={}
    for user in user_names:
        annotations=p2_results[user]
        for i,article_id in enumerate(annotations):
            annotation=annotations[article_id]
            claim_images=annotation["claim_images"]
            date=annotation['date']
            for image in claim_images:
                img_file=image.split('/')[-1]
                imgs_for_detection[img_file]+=1
                img_to_date[img_file]=date
    print ('Images for processing:',len(imgs_for_detection))
    if os.path.exists('../data/ris_result/p2_ris_results.pkl'):
        all_images=load_pkl('../data/ris_result/p2_ris_results.pkl')
        print('Already detected',len(all_images),' images')
    else:
        all_images={}
    image_list=[img_file for img_file in imgs_for_detection if img_file not in all_images]
    print ('\tRemained images for processing:',len(image_list))

    for i,img_file in enumerate(imgs_for_detection):
        #if i>10:
        #    break
        if img_file in all_images:
            print ('\tExist!',img_file)
            continue
        if i%50==0:
            print ('Already finished:',i)
            pkl.dump(all_images,open('../data/ris_result/p2_ris_results.pkl','wb'))
        img_path=os.path.join("../data/p2_annotation-1-Jan/p2_assignment_image",img_file)

        date=img_to_date[img_file]
        web_annotations=detect_web(client,img_path,args.num_queries)
        cur_raw_ris_result=[]
        print (i, 'Retrieved pages:',len(web_annotations.pages_with_matching_images))
        for page in web_annotations.pages_with_matching_images: 
            title=page.page_title
            if page.full_matching_images:
                try:
                    page_date=find_date(page.url)
                except:
                    page_date=None
            elif page.partial_matching_images:
                try:
                    page_date=find_date(page.url)
                except:
                    page_date=None
            if page_date==None or compare_date(date,page_date)==False:
                #print ('\t\tInvalid! Claim date:',date,' Page date:',page_date)
                continue
            cur_raw_ris_result.append({
                "title":title,
                "url":page.url,
                "date":page_date
            })
        print ('\t\tRemained valid:',len(cur_raw_ris_result))
        all_images[img_file]={
            'det_results': cur_raw_ris_result,
            'claim_date':date
        }
    pkl.dump(all_images,open('../data/ris_result/p2_ris_results.pkl','wb'))
    