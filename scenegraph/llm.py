import openai
import json
import requests
import time
from vl_prompt.p_manager import \
    judge_Correspondence_prompt,judge_OneObject_prompt,get_room_prompt,query_node_prompt_txt,query_node_prompt_txt_o1,target_node_prompt,query_state_transition,query_relative

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