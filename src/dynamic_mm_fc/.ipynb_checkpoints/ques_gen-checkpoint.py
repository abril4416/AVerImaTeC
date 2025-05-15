init_ques_gen_prompt="You are a fact-checker to ask questions to verify an image-text claim. "
init_ques_gen_prompt+="Here is the image-text claim. The textual part is: %s. Here is the list of images of the claim. "
init_ques_gen_prompt+="Please ask one questions related to the image part of the claim for the verification of the claim. There may be multiple images of the claim. Please explicitly point out which image you are asking about. For instance, a question could asked like:\n"
init_ques_gen_prompt+="Question: What is the event in the image?\nImage index: 3."

follow_ques_gen_prompt="You are a fact-checker to ask questions to verify an image-text claim. "
follow_ques_gen_prompt+="Here is the image-text claim. The textual part is: %s; with a list of images of the claim. We have already retrieved the evidence below: %s. [IMG] is the placeholder for images in the evidence."
follow_ques_gen_prompt+="Please ask another one question, either related to the textual part or related to the image part of the claim for the verification. For each question, please also indicate it is Text-related or Image-related before the question (using **Text-related:** and **Image-related:**). For image-related questions, please explicitly point out which image you are asking about (using **Image Index:**). For instance, a question could asked like:\n"
follow_ques_gen_prompt+="For instance: **Text-related:** [QUES]. **Image-related:** [QUES]. **Image Index:** 2.; **Image-related:** [QUES]. **Image Index:** 2,3."
