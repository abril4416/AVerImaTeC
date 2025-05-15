import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ROOT_PATH',
                        type=str,
                        default='')#remember to set your root_path here --  absolute path for AVerImaTeC project
    parser.add_argument('--LLM_NAME',
                        type=str,
                        default='qwen')#mistral
    parser.add_argument('--MLLM_NAME',
                        type=str,
                        default='qwen')#llava-next, qwen
    parser.add_argument('--MAX_QA_ITER',
                        type=int,
                        default=5)
    parser.add_argument('--NUM_GEN_QUES',
                        type=int,
                        default=8)#some MLLMs are not following instructions about the number of generated questions
    parser.add_argument('--MAX_INVALID',
                        type=int,
                        default=2)
    parser.add_argument('--SAVE_NUM',
                        type=int,
                        default=1)#0 saved for all zero-shot results/ 1 saved for all icl results
    parser.add_argument('--MAX_NUM_IMAGES', #maximum number of images in a claim
                        type=int,
                        default=3)
    parser.add_argument('--NUM_DEMOS', #number of demos in few-shot setting
                        type=int,
                        default=3)
    parser.add_argument('--ICL_FEAT',
                        type=str,
                        default='basic')#basic: BM25 for ques/textual claim; advance: BERT/CLIP for ques/image-text claims
    parser.add_argument('--TOOL_ICL',
                        type=bool,
                        default=True)
    parser.add_argument('--QG_ICL',
                        type=bool,
                        default=False)
    parser.add_argument('--DEBUG',
                        type=bool,
                        default=False)

    parser.add_argument('--PARA_QG',
                        type=bool,
                        default=False)
    parser.add_argument('--HYBRID_QG',
                        type=bool,
                        default=False)
    parser.add_argument('--GT_QUES',
                        type=bool,
                        default=False) 
    parser.add_argument('--GT_EVID',
                        type=bool,
                        default=False)#GT_EVID refer to the GT QA pairs
    parser.add_argument('--NO_SEARCH',
                        type=bool,
                        default=False)
    args=parser.parse_args()
    return args