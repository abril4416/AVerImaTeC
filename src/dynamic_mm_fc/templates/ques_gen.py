init_ques_gen_prompt_first="You are a fact-checker to ask questions to verify an image-text claim. "
init_ques_gen_prompt_first+="Here is the image-text claim. The textual part is: %s Here is the list of images of the claim: "
init_ques_gen_prompt_second="Please ask one questions related to the image part of the claim for the verification of the claim. There may be multiple images of the claim. Please explicitly point out which image you are asking about. For instance, a question could asked like: "
init_ques_gen_prompt_second+="Question: What is the event in the image? Image index: 3.\nThe image index should be smaller than the total number of claim images. Please generate your question:"

init_ques_gen_prompt={
    'first':init_ques_gen_prompt_first,
    'second':init_ques_gen_prompt_second
}

follow_ques_gen_prompt_first="You are a fact-checker to ask questions to verify an image-text claim. "
follow_ques_gen_prompt_first+="Here is the image-text claim. The textual part is: %s; with a list of images of the claim: "
follow_ques_gen_prompt_second="We have already retrieved the evidence below: %s. [IMG] is the placeholder for images in the evidence. Please ask another one question, either related to the textual part or related to the image part of the claim for the verification and avoid questions already presented in the evidence (though maybe no answer found to the question). For each question, please also indicate it is Text-related or Image-related before the question (using **Text-related:** and **Image-related:**). For image-related questions, please explicitly point out which image you are asking about (using **Image Index:**) and do not provide an index larger than the number of images (i.e., the index should be smaller than the total number of images). For instance, questions could asked like:\n"
follow_ques_gen_prompt_second+="**Text-related:** [QUES].\n**Image-related:** [QUES]. **Image Index:** 2.\n**Image-related:** [QUES]. **Image Index:** 2,3.\n[QUES] is the placeholder for the question to be generate.\nThe image index should be smaller than the total number of claim images. Please generate your question:"

follow_ques_gen_prompt={
    'first':follow_ques_gen_prompt_first,
    'second':follow_ques_gen_prompt_second
}

icl_first="You are a fact-checker to ask questions to verify an image-text claim. "
icl_first+="Please ask one questions related to the image part of the claim for the verification of the claim. There may be multiple images of the claim. Please explicitly point out which image you are asking about (the image indexing starts from 1). Below are some examples about the questions generated to verify the claim (here we only provide the textual part of the claim for these examples):\n\n%s"
icl_first+="Here is the image-text claim. The textual part is: %s; with a list of images of the claim: "

init_ques_gen_icl="\n\nPlease generate your question and please only generate one question."

icl_second="You are a fact-checker to ask questions to verify an image-text claim. "
icl_second+="Provided the claim and the evidence history, you need to ask another question to further verify the claim.  [IMG] is the placeholder for images in the evidence history. The question could be either related to the textual part or related to the image part of the claim for the verification and avoid questions already presented in the evidence (though maybe no answer found to the question). For each question, please also indicate it is Text-related or Image-related before the question (using **Text-related:** and **Image-related:**). For image-related questions, please explicitly point out which image you are asking about (using **Image Index:**) and do not provide an index larger than the number of images (i.e., the index should be smaller than the total number of images and the index starts from 1). Below are some examples about the questions generated to verify the claim (here we only provide the textual part of the claim for these examples):\n\n%s\n\nHere is the image-text claim. The textual part is: %s; with a list of images of the claim:"

follow_ques_gen_icl=" Evidence history: %s Generated question:"

icl_ques_gen_prompt={
    'first':icl_first,
    'init_ques':init_ques_gen_icl,
    'second':icl_second,
    'follow_ques':follow_ques_gen_icl
}


para_ques_prompt="Please ask %s questions related to the textual part or the image part of the claim for the verification of the claim. "
para_ques_prompt+="For each question, please also indicate it is Text-related or Image-related before the question. "
para_ques_prompt+="For Image-related questions, please also indicate which images the question is about with **Image Index:**. Specifically, if the question is related to the first and second image of the claim, it should be **Image Index:** 1,2.\n"
para_ques_prompt+="We illustrate an example of first three questions as below ([QUES] is the placeholder for the question.):\n\n1.**Text-related:** [QUES]\n2. **Image-related:** [QUES] **Image Index:** 1.\n3. **Image-related:** [QUES] **Image Index:** 2.\n\nPlease generate %s questions:"

para_ques_prompt_icl="Please ask %s questions related to the textual part or the image part of the claim for the verification of the claim. "
para_ques_prompt_icl+="For each question, please also indicate it is Text-related or Image-related before the question. "
para_ques_prompt_icl+="For Image-related questions, please also indicate which images the question is about with **Image Index:**.\n\n%s\n\nPlease generate %s questions:"