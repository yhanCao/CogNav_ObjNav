import openai
import json
import requests
import time
from vl_prompt.p_manager import extract_integer_answer, extract_scores, extract_objects, \
    get_frontier_prompt, get_candidate_prompt, get_grouping_prompt, get_discover_prompt,get_relationship_prompt,get_one_relation_prompt,\
    judge_Correspondence_prompt,judge_OneObject_prompt,discover_OneObject_prompt,get_room_prompt,query_node_prompt,query_node_prompt_txt,query_node_prompt_txt_o1,target_node_prompt,query_state_transition,query_relative

# should be replaced by your API key
with open("./apikey.txt") as f:
    # key order：temp key, long-term, mine
    keys = f.read().split("\n")
    openai.api_key = keys[0] # NOTE: change key before running


class LLM():
    def __init__(self, goal_name, prompt_type):
        self.api_name = "gpt-4o"
        # self.api_name = "gpt-3.5-turbo"
        self.goal_name = goal_name
        self.prompt_type = prompt_type
        self.history = []
    
    def inference_once(self, system_prompt, message):
        if message:
            msg = {
                "role": "user",
                "content": message
            }
            self.history = system_prompt + [msg]
            try:
                chat = openai.ChatCompletion.create(
                    model=self.api_name, messages=self.history,
                    temperature=0
                )
            except Exception as e:
                print(f"=====> llm inference error: {e}")
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=self.history,
                    temperature=0
                )
        reply = chat.choices[0].message.content
        return reply
    
    def discover_objects(self, img, objects):
        payload = get_discover_prompt(img, objects)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        reply = response.json()["choices"][0]["message"]["content"]
        c = extract_objects(reply)
        return c
    def get_room(self, img):
        payload = get_room_prompt(img)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # print(response.json())
        reply = response.json()["choices"][0]["message"]["content"]
        return reply
    def target_query(self, img):
        payload = get_room_prompt(img)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # print(response.json())
        reply = response.json()["choices"][0]["message"]["content"]
        return reply
    def query_target_obj(self,img,target_name):
        user=f"Is {target_name} in this image ?\n"
        payload = target_node_prompt(img,user)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        return reply
    def query_node_graph(self,img,user):
        payload = query_node_prompt(img,user)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        print(reply[reply.find(":"):reply.find("\n")])
        return reply,reply[reply.find("[")+1:reply.find("]")]
    def query_node_txt_o1(self,user):
        client=openai.OpenAI(api_key=openai.api_key)
        prompt = query_node_prompt_txt_o1(user)
        
        response = client.chat.completions.create(model="o1-preview", messages=[{
            "role": "user", "content": [
                {"type": "text", "text": prompt+user}
            ]
        }])

        reply = response.choices[0].message.content
        print(reply[reply.find(":"):reply.find("\n")])
        return reply,reply[reply.find("[")+1:reply.find("]")]
    def query_node_txt(self,user):
        payload = query_node_prompt_txt(user)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        print(reply[reply.find(":"):reply.find("\n")])
        return reply,reply[reply.find("[")+1:reply.find("]")]
    def query_state_transition(self,user):
        payload = query_state_transition(user)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        print(reply[reply.find(":"):reply.find("\n")])
        return reply,reply[reply.find("[")+1:reply.find("]")]
    def query_state_transition(self,user):
        payload = query_relative(user)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        print(reply[reply.find(":"):reply.find("\n")])
        return reply,reply[reply.find("[")+1:reply.find("]")]
    def relation_of_objects(self,img,objects):
        payload = get_discover_prompt(img, objects)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        return reply
    def inference_accumulate(self, message):
        if message:
            self.history.append(
                {"role": "user", "content": message},
            )
            try:
                chat = openai.ChatCompletion.create(
                    model=self.api_name, messages=self.messages
                )
            except Exception as e:
                print(f"=====> gpt-4-turbo error: {e}")
                chat = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", messages=self.history,
                    temperature=0
                )
            
        reply = chat.choices[0].message.content
        # print(reply)
        self.history.append({"role": "assistant", "content": reply})
        return reply
    
    def choose_frontier(self, message):
        system_prompt = get_frontier_prompt(self.prompt_type)
        reply = self.inference_once(system_prompt, message)
        if self.prompt_type == "deterministic":
            answer = extract_integer_answer(reply)
        elif self.prompt_type == "scoring":
            answer, scores = extract_scores(reply)
        
        return answer, reply
        
    def imagine_candidate(self, instr):
        system_prompt = get_candidate_prompt(candidate_type="open")
        reply = self.inference_once(system_prompt, instr)
        c = extract_objects(reply)
        return c
    
    def group_candidate(self, clist, nlist):
        system_prompt = get_grouping_prompt()
        message = f"Current object list: {clist}\n\nNew object list: {nlist}"
        reply = self.inference_once(system_prompt, message)
        c = extract_objects(reply) # newly discovered object list
        new_c = []
        for ob in c:
            if ("room" in ob) or ("wall" in ob) or ("floor" in ob) or ("ceiling" in ob):
                continue
            new_c.append(ob)
        return new_c
    def get_relationship_prompt(self,object,olist,image):
        payload = get_relationship_prompt(object,olist,image)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        c=extract_objects(reply)
        print(object,olist,c)
        return c
    def get_relationship2_prompt(self,bboxfile,relation_candidate,image_files):
        with open(bboxfile, "r") as f:
            bbox = json.load(f)
        with open(relation_candidate, "r") as f:
            relation = json.load(f)
        with open(image_files, "r") as f:
            images = json.load(f)
        for object,olist in relation.items():
            start=time.time()
            print(object,olist)
            payload = get_relationship_prompt(bbox, object,olist,images[object])
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            print(response.json())
            reply = response.json()["choices"][0]["message"]["content"]
            odict={}
            c=extract_objects(reply)
            end=time.time()
            print("per query time:",end-start)
            if len(c) == len(olist):
                for i in range(len(olist)):
                    odict[olist[i]]=c[i]
                with open("relationship/"+str(object)+".json", "w") as f:
                    json.dump(odict, f, indent=4)
    def get_one_relation_prompt(self,bboxfile,objectlist,image_file):
        with open(bboxfile, "r") as f:
            bbox = json.load(f)
        payload = get_one_relation_prompt(bbox,objectlist,image_file)
        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        print(reply[reply.find(":")+1:])
        r=json.loads(reply[reply.find(":")+1:])
        return r
    def jduge_prompt(self,bboxfile,objectlist,image_files):
        with open(bboxfile, "r") as f:
            bbox = json.load(f)
        payload = judge_Correspondence_prompt(bbox,objectlist,image_files)
        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        print(reply[reply.find(":")+1:])
        return reply
    
    def jduge_One_prompt(self,objectlist,objects,image_file):
        payload = judge_OneObject_prompt(objectlist,objects,image_file)
        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        print(reply)
        return reply
    def discover_One_prompt(self,objectlist,image_file):
        payload = discover_OneObject_prompt(objectlist,image_file)
        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai.api_key}"
            }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        print(reply)
        return reply[reply.find(":")+1:]
    