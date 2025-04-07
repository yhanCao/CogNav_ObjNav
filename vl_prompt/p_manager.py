import re
from .utils import encode_image_gpt4v
from PIL import Image
import base64
import json
from io import BytesIO
def extract_integer_answer(s):
    match = re.search(r'Answer: \d+', s)
    if match:
        return int(match.group().split(' ')[-1])
    else:
        print('=====> No integer found in string')
        return -1
    
    
def extract_scores(s):
    match = re.search(r'Answer: \[(.*?)\]', s)
    if match:
        scores = [float(x) for x in match.group(1).split(',')]
        return scores.index(max(scores)), scores
    else:
        print('=====> No list found in string')
        return -1, []
    
def extract_objects(s):
    elements = re.findall(r'"([^"]*)"', s)
    return list(elements)
    

def object_query_constructor(objects):
    """
    Construct a query string based on a list of objects

    Args:
        objects: torch.tensor of object indices contained in an area

    Returns:
        str query describing the area, eg "This area contains toilet and sink."
    """
    assert len(objects) > 0
    query_str = "This area contains "
    names = []
    for ob in objects:
        names.append(ob.replace("_", " "))
    if len(names) == 1:
        query_str += names[0]
    elif len(names) == 2:
        query_str += names[0] + " and " + names[1]
    else:
        for name in names[:-1]:
            query_str += name + ", "
        query_str += "and " + names[-1]
    query_str += "."
    return query_str


def get_room_prompt(img_name):
    from vl_prompt.prompt_cog.roomJudge import \
        SYSTEM_PROMPT, USER
    with open(img_name, "rb") as image_file:
        img = base64.b64encode(image_file.read()).decode('utf-8')
    # question = f"""Current object list: {objects}\n{USER}"""
    payload = {
        "model": "gpt-4o", "messages": [{
            "role": "system", "content": SYSTEM_PROMPT
            }, {
            "role": "user", "content": [
                {"type": "text", "text": USER}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            ]
        }], "max_tokens": 300
    }
    return payload
def target_node_prompt(img_name,user):
    from vl_prompt.prompt_gpt.target_object_makesure import \
        SYSTEM_PROMPT
    with open(img_name, "rb") as image_file:
        img = base64.b64encode(image_file.read()).decode('utf-8')
    # question = f"""Current object list: {objects}\n{USER}"""
    payload = {
        "model": "gpt-4o", "messages": [{
            "role": "system", "content": SYSTEM_PROMPT
            }, {
            "role": "user", "content": [
                {"type": "text", "text": user}, 
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
            ]
        }], "max_tokens": 300
    }
    return payload
    
def query_node_prompt_txt_o1(user):
    from vl_prompt.prompt_gpt.queryNodetxt import \
        SYSTEM_PROMPT
    payload = {
        "model": "o1-preview", "messages": [{
            "role": "system", "content": SYSTEM_PROMPT
            }, {
            "role": "user", "content": [
                {"type": "text", "text": user}
            ]
        }], "max_tokens": 800
    }
    return SYSTEM_PROMPT
def query_node_prompt_txt(user):
    from vl_prompt.prompt_gpt.queryNodetxt import \
        SYSTEM_PROMPT
    # question = f"""Current object list: {objects}\n{USER}"""
    payload = {
        "model": "gpt-4o", "messages": [{
            "role": "system", "content": SYSTEM_PROMPT
            }, {
            "role": "user", "content": [
                {"type": "text", "text": user}
            ]
        }], "max_tokens": 800
    }
    return payload
def query_state_transition(user):
    from vl_prompt.prompt_gpt.state_transition import \
        SYSTEM_PROMPT
    # question = f"""Current object list: {objects}\n{USER}"""
    payload = {
        "model": "gpt-4o", "messages": [{
            "role": "system", "content": SYSTEM_PROMPT
            }, {
            "role": "user", "content": [
                {"type": "text", "text": user}
            ]
        }], "max_tokens": 800
    }
    return payload
def query_relative(user):
    from vl_prompt.prompt_gpt.query_relative import \
        SYSTEM_PROMPT
    # question = f"""Current object list: {objects}\n{USER}"""
    payload = {
        "model": "gpt-4o", "messages": [{
            "role": "system", "content": SYSTEM_PROMPT
            }, {
            "role": "user", "content": [
                {"type": "text", "text": user}
            ]
        }], "max_tokens": 800
    }
    return payload


def judge_Correspondence_prompt(bbox,list,image_urls):
    from vl_prompt.prompt_gpt.judgeCorrespondence import \
        SYSTEM_PROMPT, USER
    message=[{"role": "system", "content": SYSTEM_PROMPT}]
    question = f"""{USER}" "{list}"""
    user_content =[ {  "type": "text", "text": question+f"\n{json.dumps(bbox,indent=2)}"  }]
    for image in image_urls :
        user_content.append({ "type": "image_url", "image_url": { "url": encode_image_gpt4v(image)} } )
    message.append( {"role": "user", "content": user_content})
    payload = {
        "model": "gpt-4-vision-preview", "messages": message, "max_tokens": 300
    }
    return payload

def judge_OneObject_prompt(objectlist,objects,image_url):
    from vl_prompt.prompt_gpt.judgeOneobject import \
        SYSTEM_PROMPT, USER
    message=[{"role": "system", "content": SYSTEM_PROMPT}]
    question = f"""{USER}" "{objectlist}" "{objects} """
    user_content =[ {  "type": "text", "text": question},{ "type": "image_url", "image_url": { "url": encode_image_gpt4v(image_url)} }]
    message.append( {"role": "user", "content": user_content})
    payload = {
        "model": "gpt-4-vision-preview", "messages": message, "max_tokens": 300
    }
    return payload
