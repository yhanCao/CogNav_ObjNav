import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
import json

# 从JSON文件读取数据并转换为字典
# with open('css4_colors.json', 'r', encoding='utf-8') as f:
#     css4_colors_dict = json.load(f)
relationship_candidate=["next to","on","in","hanging on","none"]
def load_model(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    return tokenizer,model
def find_element_in_string(string, elements):
    for element in elements:
        if element in string:
            return element
    return None

def COG_one(image_path,object_ids,tokenizer,model):
    from vl_prompt.prompt_cog.relationship_one import SYSTEM_PROMPT,USER1,USER2
    system = SYSTEM_PROMPT 
    image = Image.open(image_path).convert('RGB')
    relationship={}
    for pair in object_ids:
        user_prompt=USER1+str(pair[0]) + USER2 + str(pair[1])
        query = user_prompt + system
        input_by_model = model.build_conversation_input_ids(
                    tokenizer,
                    query=query,
                    images=[image],
                    template_version='chat'
                )
        inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
            }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            relation=find_element_in_string(response[response.find('is'):],relationship_candidate)
            if pair[0] in relationship.keys() :
                relationship[pair[0]][pair[1]]=relation
            else :
                rr={pair[1]:relation}
                relationship[pair[0]]=rr
    return relationship

def judgeRoom(image,tokenizer,model):
    from vl_prompt.prompt_cog.roomJudge import SYSTEM_PROMPT,USER
    query = SYSTEM_PROMPT
    input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                images=[image],
                template_version='chat'
            )
    inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]
    return response
