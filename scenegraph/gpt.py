import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd

from llm import LLM

os.environ["OMP_NUM_THREADS"] = "1"

def RelationVLM(objects,image):
    relation={}
    cname="chair"
    lm = LLM(cname,"scoring")
    for object,olist in objects.items():
        relationone=lm.get_relationship_prompt(object,olist,image)
        rr={}
        for i in range(len(olist)):
            rr[olist[i]]=relationone[i]
        relation[object]=rr
    return relation
