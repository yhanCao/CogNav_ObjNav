# SYSTEM_PROMPT="""
# The input includes a object list and an object, you need to discover all objects in the list and the second input object, and then analyze semantic information of the input image to give the relationship.
# The relationship must select one from the following 5 elements: ["next to","on","in","hanging on","none"]. 
# You must only output the relationship and nothing else.
# Your output relationship should include all objects in the first input list in the form of "Answer": {<object id>:<relationship>} and nothing else, such as:
# Answer: {2:on,3:next to}
# """
# SYSTEM_PROMPT="""You need to analyze semantic information of the input image and object tag to give the relationship.
# The relationship must select one from the following 5 elements: ["next to","on","in","hanging on","none"], You must only output the relationship and nothing else. 
# """
# USER1="""What are the relationship of """
# USER2=""" and """

SYSTEM_PROMPT=""" The input is multiple pairs of object ids, in form of [<id1>,<id2>],[<id1>,<id2>]...
The object <id> in image with color mask represents an object with tag <id>.
You need to analyze semantic information of the image and input object tags to give the <relationship>.
The relationship must be one of the following 6 elements: ["next to","on","in","under","hanging on","none"] and nothing else, which has direction and represents <id1> is <relationship> <id2>.
Each input pair corresponds to a <relationship> of object id1 and object id2, with output format "<id1> <relationship> <id2>".
You must output a list of "<id1> <relationship> <id2>" for every input pair of object ids, as format like :
1. [<id1>,<id2>] : "on"
2. [<id1>,<id2>] : "next to "
3. [<id1>,<id2>] : "none"
Plese do not output the analysis of the process.
"""
USER="""What are the relationship of these pairs? """

# You should consider the following rules when discovering the relationship:
# (1)
# (2) You need to analyze semantic information of the input image to give the relationship.
# "on" : if one object in key "id" is an object commonly placed on top of the another one.
# "in" : if one object in key "id" is an object commonly placed inside the another one.
# "next to" : if two objects position in the same plane, close and parallel to each other, and with no remaining objects between them.
# "hanging on" : if one object in key "id" hold onto the another one firmly to avoid falling or losing grip.
# "none" : if none of the above best describe the relationship between the two objects.
# You need to only output a relationship (and nothing else). such as: "on", and don't output the reason.

# You must only output the relationship and nothing else in the form of "Answer": [relationship1,relationship2...] and nothing else, such as:
# Answer: [on,next to,none] \n