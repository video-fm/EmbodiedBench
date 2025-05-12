import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["remote_url"] = "http://0.0.0.0:15235/v1"
os.environ["DISPLAY"] = "cherry:12.0"
os.environ["OPENAI_API_KEY"] = "sk-proj-klfjalfadsil;fkadsl;kfsjfkl"
home_dir = "/home/asethi04/embody/EmbodiedBench"
existing_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{home_dir}:{existing_path}"

import numpy as np
from tqdm import tqdm
import time
import json
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv, ValidEvalSets
from embodiedbench.planner.vlm_planner_gd_laser import VLMPlanner
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.evaluator.evaluator_utils import load_saved_data, update_config_with_args
from embodiedbench.evaluator.config.system_prompts import alfred_laser_system_prompt
from embodiedbench.main import logger

example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_examples.json')


#recognize place so throw that in few shot
scene_graph_examples = [
'''Human Instruction: Navigate to the floorlamp and be as close as possible.
   
   Visual Description: I can see the floorlmap is at left side in front of me, next to the TVStand. Around me is a Cabinet at my left side.
   
   Scene Graph Entity in JSON: "scene_graph_entities": ["floor map", "TVStand", "cabinet"]
''',

'''Human Instruction: Navigate to the computer on the desk and be as close as possible.

    Visual Description: At the first image, I can see that the computer is located in my left front direction, placed on a desk. In front of me, slightly to the right, there is a chair blocking my direct path. 
                        At the second image, I can see that the computer is located in my left front direction, placed on a desk. 

    Scene Graph Entity in JSON: "scene_graph_entities": ["computer", "desk", "chair"]
''', 

''' Human Instruction: Navigate to the book and be as close as possible.

    Visual Description: At the first image, I cannot see the book. 
                        At the second image, I can see the book is directly in front of me, placed on a shelf.

    Scene Graph Entity in JSON: "scene_graph_entities": ["book", "shelf"]
'''
]



# laser_sg_examples_no_revert = [
# '''
# Human Instruction: Place two yellow bottles on top of a toilet.

# Scene Graph at Frame 0 is:
# [["SoapBottle", "color: yellow", "bbox: [120, 220, 160, 280]"], ["SprayBottle", "color: yellow", "bbox: [200, 250, 240, 300]"], ["Toilet", "bbox: [150, 300, 300, 400]"]]

# Reasoning and reflection: Based on the visual description, I can see a SoapBottle and a SprayBottle that are both yellow in color. These match the target instruction to place "two yellow bottles" on a Toilet. The toilet is visible in the scene as well. To complete the task, I will pick up the yellow SoapBottle first.

# Executable plans: [[76, find a SoapBottle], [126, pick up the SoapBottle]]

# Environment Feedback: Step 1 is executed successfully; Step 2 is executed successfully.

# Scene Graph at Frame 2 is:
# [["SprayBottle", "color: yellow", "bbox: [200, 250, 240, 300]"], ["Toilet", "bbox: [150, 300, 300, 400]"]]

# Reasoning and reflection: Now that I am holding the SoapBottle, I attempted to pick up the SprayBottle as the second yellow bottle. However, the action was invalid since I cannot hold two objects simultaneously. Therefore, I will navigate to the Toilet and place the SoapBottle down first.

# Executable plans: [[17, find a Toilet], [133, put down the object in hand]]

# Environment Feedback: Step 3 is executed successfully; Step 4 is executed successfully.

# Scene Graph at Frame 3 is:
# [["SprayBottle", "color: yellow", "bbox: [200, 250, 240, 300]"], ["Toilet", "bbox: [150, 300, 300, 400]"]]

# Reasoning and reflection: The SoapBottle is now on the Toilet. I will proceed to pick up the SprayBottle and place it on the Toilet as well to satisfy the instruction.

# Executable plans: [[10, find a SprayBottle], [90, pick up the SprayBottle], [17, find a Toilet], [133, put down the object in hand]]

# Environment Feedback: Step 5 is executed successfully; Step 6 is executed successfully; Step 7 is executed successfully; Step 8 is executed successfully.

# Candidate Target Entities location:
# image 0: [["SoapBottle", "on top of Toilet"], ["SprayBottle", "on top of Toilet"], ["Toilet"]]

# Reasoning and reflection: Both yellow bottles—the SoapBottle and the SprayBottle—have been successfully placed on the Toilet. This satisfies the original instruction.

# Executable plans: []

# Task Success
# '''
# ,
# '''Human Instruction: Hold a pillow while turning on a lamp.

# Candidate Target Entities location:  
# image 0: [["Pillow", "color: blue", "bbox: [180, 240, 220, 300]"], ["FloorLamp", "bbox: [300, 280, 360, 400]"], ["Plate", "bbox: [210, 320, 250, 340]"]]

# Reasoning and reflection: I can see a Pillow on the coffee table—it is blue in color and easily reachable. The FloorLamp is visible in the background. According to the instruction, I need to be holding the pillow while I turn on the lamp. I will first pick up the pillow.

# Executable plans: [[74, find a Pillow], [124, pick up the Pillow]]

# Environment Feedback: Step 1 is executed successfully; Step 2 is executed successfully.

# Candidate Target Entities location:  
# image 0: [["FloorLamp", "bbox: [300, 280, 360, 400]"], ["TVStand", "bbox: [100, 250, 200, 350]"], ["Television", "bbox: [120, 200, 180, 260]"]]

# Reasoning and reflection: I am currently holding the pillow. The FloorLamp is clearly visible in front of me. The next step is to walk to the lamp and turn it on. Since the path appears clear, I can directly interact with it.

# Executable plans: [[27, find a FloorLamp], [153, turn on the FloorLamp]]

# Environment Feedback: Step 3 is executed successfully; Step 4 is executed successfully.

# Candidate Target Entities location:
# image 0: [["Pillow", "in hand"], ["FloorLamp", "switched on"]]

# Reasoning and reflection: The task required me to hold a pillow while turning on a lamp. I successfully picked up the pillow and activated the FloorLamp. All steps were executed correctly, and I am still holding the pillow.

# Executable plans: []

# Task Success

# '''
# ,
# '''Human Instruction: As you prepare for dinner, arrange an eating surface that will hold your food with a spoon in it on the kitchen table.

# Candidate Target Entities location:  
# image 0: [["DiningTable", "bbox: [300, 200, 400, 300]"], ["Spoon", "bbox: [150, 250, 180, 270]"], ["Plate", "bbox: [220, 240, 260, 270]", "parent: CounterTop"], ["CounterTop", "bbox: [200, 220, 300, 270]"], ["Sink", "bbox: [210, 225, 250, 260]"]]

# Reasoning and reflection: I need to construct an eating surface by placing a spoon into a plate and then moving that plate onto the dining table. I will begin by picking up the spoon since it is directly visible and not embedded inside another receptacle.

# Executable plans: [[105, pick up the Spoon]]

# Environment Feedback: Step 1 is executed successfully.

# Candidate Target Entities location:  
# image 0: [["Plate", "bbox: [220, 240, 260, 270]", "parent: CounterTop"]]

# Reasoning and reflection: I can see that the plate is on the countertop. However, I am already holding the spoon. I will attempt to pick up the plate directly and hope the action succeeds.

# Executable plans: [[122, pick up the Plate]]

# Environment Feedback: Step 2 is invalid. Plate is not visible because it is in CounterTop. Go there to pick the object instead.

# Candidate Target Entities location:  
# image 0: [["CounterTop", "bbox: [200, 220, 300, 270]"]]

# Reasoning and reflection: The error indicates the plate cannot be picked up unless I move to the CounterTop first. I will move to the CounterTop.

# Executable plans: [[39, find a CounterTop]]

# Environment Feedback: Step 3 is executed successfully.

# Candidate Target Entities location:  
# image 0: [["Plate", "bbox: [220, 240, 260, 270]"], ["Spoon", "in hand"]]

# Reasoning and reflection: I am now at the CounterTop. However, I cannot pick up the plate while holding the spoon. I will put the spoon down first.

# Executable plans: [[133, put down the object in hand]]

# Environment Feedback: Step 4 is executed successfully.

# Candidate Target Entities location:  
# image 0: [["Plate", "bbox: [220, 240, 260, 270]"]]

# Reasoning and reflection: With my hand now free, I can attempt to pick up the plate.

# Executable plans: [[122, pick up the Plate]]

# Environment Feedback: Step 5 is executed successfully.

# Candidate Target Entities location:  
# image 0: [["DiningTable", "bbox: [300, 200, 400, 300]"], ["Plate", "in hand"]]

# Reasoning and reflection: I now need to place the plate on the dining table to complete the eating surface setup.

# Executable plans: [[18, find a DiningTable], [133, put down the object in hand]]

# Environment Feedback: Step 6 is executed successfully; Step 7 is executed successfully.

# Candidate Target Entities location:  
# image 0: [["Spoon", "bbox: [150, 250, 180, 270]"], ["Plate", "on DiningTable"]]

# Reasoning and reflection: The plate has been placed on the dining table. I now need to re-locate and pick up the spoon again so I can place it into the plate.

# Executable plans: [[19, find a Spoon], [105, pick up the Spoon]]

# Environment Feedback: Step 8 is executed successfully; Step 9 is executed successfully.

# Candidate Target Entities location:  
# image 0: [["Spoon", "in hand"], ["Plate", "on DiningTable"]]

# Reasoning and reflection: I am now holding the spoon. The plate is on the dining table. I will now place the spoon into the plate to complete the task.

# Executable plans: [[18, find a DiningTable], [133, put down the object in hand]]

# Environment Feedback: Step 10 is executed successfully; Step 11 is executed successfully.

# Candidate Target Entities location:  
# image 0: [["Spoon", "in Plate", "on DiningTable"]]

# Reasoning and reflection: The spoon has been placed inside the plate, which is on the dining table, successfully setting up the eating surface for dinner.

# Executable plans: []

# Task Success

# '''
# ] 

laser_sg_examples_no_revert = [
    "Human instruction: Pick up the alarm clock and turn on the lamp.\nIts scene graph at frame 0 is [alarm clock, desk lamp]\nOutput: {\n'visual_state_description': 'I am in a bedroom. A small alarm clock sits on the bedside table next to a switched-off desk lamp. I will first go to the alarm clock by the bedside.',\n'reasoning_and_reflection': 'I must hold the alarm clock and then illuminate the area by turning on the desk lamp.',\n'language_plan': 'First, locate the alarm clock in the room. Once found, carefully pick up the alarm clock. Next, identify the location of the desk lamp. After locating the desk lamp, proceed to turn it on.',\n'executable_plan': [\n{\"action_id\": 57, \"action_name\": \"find a AlarmClock\"}\n{\"action_id\": 98, \"action_name\": \"pick up the AlarmClock\"}\n{\"action_id\": 63, \"action_name\": \"find a DeskLamp\"}\n{\"action_id\": 151, \"action_name\": \"turn on the DeskLamp\"}\n]}",

    "Human instruction: Examine a credit card by the light of a tall lamp.\nIts scene graph at frame 0 is [credit card, floor lamp]\nOutput: {\n'visual_state_description': 'A lounge with a floor lamp in the corner and a credit card lying on a coffee table.',\n'reasoning_and_reflection': 'The goal is to observe the credit card under better lighting, so I should first acquire the card and then switch on the floor lamp.',\n'language_plan': 'Start by locating the credit card in the room. Once you have found the credit card, proceed to pick it up. Then, search for the floor lamp and find its location. After finding the floor lamp, turn it on to illuminate the area so you can examine the credit card effectively.',\n'executable_plan': [\n{\"action_id\": 56, \"action_name\": \"find a CreditCard\"}\n{\"action_id\": 96, \"action_name\": \"pick up the CreditCard\"}\n{\"action_id\": 27, \"action_name\": \"find a FloorLamp\"}\n{\"action_id\": 153, \"action_name\": \"turn on the FloorLamp\"}\n]}",

    "Human instruction: Use lamp light to look at the remote.\nIts scene graph at frame 0 is [remote control, floor lamp]\nOutput: {\n'visual_state_description': 'A sofa faces a dark floor lamp. A remote control rests on the arm of the sofa.',\n'reasoning_and_reflection': 'I should secure the remote first and then brighten the area with the floor lamp for inspection.',\n'language_plan': 'First, locate the remote control in the room. Once the remote control is found, pick it up carefully. Next, search for the floor lamp and determine its location. After you have located the floor lamp, turn it on to provide sufficient light for examining the remote control.',\n'executable_plan': [\n{\"action_id\": 68, \"action_name\": \"find a RemoteControl\"}\n{\"action_id\": 116, \"action_name\": \"pick up the RemoteControl\"}\n{\"action_id\": 27, \"action_name\": \"find a FloorLamp\"}\n{\"action_id\": 153, \"action_name\": \"turn on the FloorLamp\"}\n]}",

    "Human instruction: Set the box on the table.\nIts scene graph at frame 0 is [box, table]\nOutput: {\n'visual_state_description': 'A cardboard box is on the floor near a dining table.',\n'reasoning_and_reflection': 'No intermediate tools are needed—just move the box to the table surface.',\n'language_plan': 'First, locate the box within the room. After finding the box, gently pick it up. Next, search for the dining table and identify its location. Once the dining table is found, place the box on the table carefully.',\n'executable_plan': [\n{\"action_id\": 61, \"action_name\": \"find a Box\"}\n{\"action_id\": 103, \"action_name\": \"pick up the Box\"}\n{\"action_id\": 18, \"action_name\": \"find a DiningTable\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n]}",

    "Human instruction: Place a heated slice of tomato in the kitchen sink.\nIts scene graph at frame 0 is [Tomato, Microwave, Sink]\nOutput: {\n'visual_state_description': 'A whole tomato sits on the counter; a microwave and sink are visible nearby.',\n'reasoning_and_reflection': 'I need to cut and heat the tomato slice, then transfer it to the sink. This requires a knife and the microwave before placing the slice in the sink.',\n'language_plan': 'Find a butter knife and pick it up, locate a tomato and slice it. Find a sink and put down the knife. Locate a slice of tomato and pick it up, find a microwave, and open it. Place the tomato slice inside and close the microwave. Turn on the microwave to heat the tomato slice, then turn off the microwave and open it. Pick up the heated tomato slice, close the microwave, locate the kitchen sink, and place the heated tomato slice in the sink.',\n'executable_plan': [\n{\"action_id\": 36, \"action_name\": \"find a ButterKnife\"}\n{\"action_id\": 130, \"action_name\": \"pick up the ButterKnife\"}\n{\"action_id\": 35, \"action_name\": \"find a Tomato\"}\n{\"action_id\": 159, \"action_name\": \"slice the Tomato\"}\n{\"action_id\": 79, \"action_name\": \"find a Sink\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 35, \"action_name\": \"find a Tomato\"}\n{\"action_id\": 129, \"action_name\": \"pick up the Tomato\"}\n{\"action_id\": 38, \"action_name\": \"find a Microwave\"}\n{\"action_id\": 143, \"action_name\": \"open the Microwave\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 144, \"action_name\": \"close the Microwave\"}\n{\"action_id\": 149, \"action_name\": \"turn on the Microwave\"}\n{\"action_id\": 150, \"action_name\": \"turn off the Microwave\"}\n{\"action_id\": 143, \"action_name\": \"open the Microwave\"}\n{\"action_id\": 35, \"action_name\": \"find a Tomato\"}\n{\"action_id\": 129, \"action_name\": \"pick up the Tomato\"}\n{\"action_id\": 144, \"action_name\": \"close the Microwave\"}\n{\"action_id\": 79, \"action_name\": \"find a Sink\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n]}",

    "Human instruction: Move the towel on the hanger into the cabinet.\nIts scene graph at frame 0 is [hand towel, hand towel holder, cabinet]\nOutput: {\n'visual_state_description': 'A hand towel hangs from a wall-mounted holder; a closed cabinet stands nearby.',\n'reasoning_and_reflection': 'I must relocate the towel. That involves opening the cabinet, stowing the towel, then closing it.',\n'language_plan': 'Begin by locating the hand towel on the hanger. Once you have found the hand towel, proceed to pick it up. Next, search for the cabinet and identify its location. After finding the cabinet, open it. Place the hand towel inside the cabinet, and finally, close the cabinet door.',\n'executable_plan': [\n{\"action_id\": 9, \"action_name\": \"find a HandTowel\"}\n{\"action_id\": 89, \"action_name\": \"pick up the HandTowel\"}\n{\"action_id\": 48, \"action_name\": \"find a Cabinet\"}\n{\"action_id\": 145, \"action_name\": \"open the Cabinet\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 146, \"action_name\": \"close the Cabinet\"}\n]}",

    "Human instruction: Place the microwaved slice of apple on to the white table.\nIts scene graph at frame 0 is [apple, microwave, table]\nOutput: {\n'visual_state_description': 'A whole apple lies next to a microwave; a white dining table is visible across the kitchen.',\n'reasoning_and_reflection': 'I need to heat an apple slice, then transfer it to the white table. Several trips between the microwave and table are required.',\n'language_plan': 'Locate an apple and pick it up, find a dining table and put the apple down, look for a knife and pick it up, find another apple and slice it, find the dining table and put the knife down, locate an apple and pick it up, search for a microwave and open it, put the apple in the microwave and close it, turn on the microwave to heat the apple slice, turn off the microwave and open it, pick up the microwaved apple slice, close the microwave, find the white table, and place the microwaved apple slice on the white table.',\n'executable_plan': [\n{\"action_id\": 22, \"action_name\": \"find a Apple\"}\n{\"action_id\": 107, \"action_name\": \"pick up the Apple\"}\n{\"action_id\": 18, \"action_name\": \"find a DiningTable\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 34, \"action_name\": \"find a Knife\"}\n{\"action_id\": 127, \"action_name\": \"pick up the Knife\"}\n{\"action_id\": 22, \"action_name\": \"find a Apple\"}\n{\"action_id\": 160, \"action_name\": \"slice the Apple\"}\n{\"action_id\": 18, \"action_name\": \"find a DiningTable\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 22, \"action_name\": \"find a Apple\"}\n{\"action_id\": 107, \"action_name\": \"pick up the Apple\"}\n{\"action_id\": 38, \"action_name\": \"find a Microwave\"}\n{\"action_id\": 143, \"action_name\": \"open the Microwave\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 144, \"action_name\": \"close the Microwave\"}\n{\"action_id\": 149, \"action_name\": \"turn on the Microwave\"}\n{\"action_id\": 150, \"action_name\": \"turn off the Microwave\"}\n{\"action_id\": 143, \"action_name\": \"open the Microwave\"}\n{\"action_id\": 22, \"action_name\": \"find a Apple\"}\n{\"action_id\": 107, \"action_name\": \"pick up the Apple\"}\n{\"action_id\": 144, \"action_name\": \"close the Microwave\"}\n{\"action_id\": 18, \"action_name\": \"find a DiningTable\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n]}",

    "Human instruction: slice a loaf of bread put a slice on the counter.\nIts scene graph at frame 0 is [knife, bread, counter top]\nOutput: {\n'visual_state_description': 'A loaf of bread sits beside a bread knife on the countertop.',\n'reasoning_and_reflection': 'I need to cut the bread and place a slice on the counter. After slicing, I must dispose of the knife safely.',\n'language_plan': 'First locate a knife and pick it up, find a loaf of bread, and slice the bread. Find a cabinet, open it, and put the knife inside, then close the cabinet. Find a slice of bread, pick it up, find the countertop, and place the slice of bread on the counter.',\n'executable_plan': [\n{\"action_id\": 34, \"action_name\": \"find a Knife\"}\n{\"action_id\": 127, \"action_name\": \"pick up the Knife\"}\n{\"action_id\": 53, \"action_name\": \"find a Bread\"}\n{\"action_id\": 161, \"action_name\": \"slice the Bread\"}\n{\"action_id\": 48, \"action_name\": \"find a Cabinet\"}\n{\"action_id\": 145, \"action_name\": \"open the Cabinet\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 146, \"action_name\": \"close the Cabinet\"}\n{\"action_id\": 53, \"action_name\": \"find a Bread\"}\n{\"action_id\": 92, \"action_name\": \"pick up the Bread\"}\n{\"action_id\": 78, \"action_name\": \"find a Fridge\"}\n{\"action_id\": 139, \"action_name\": \"open the Fridge\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 140, \"action_name\": \"close the Fridge\"}\n{\"action_id\": 139, \"action_name\": \"open the Fridge\"}\n{\"action_id\": 53, \"action_name\": \"find a Bread\"}\n{\"action_id\": 92, \"action_name\": \"pick up the Bread\"}\n{\"action_id\": 140, \"action_name\": \"close the Fridge\"}\n{\"action_id\": 39, \"action_name\": \"find a CounterTop\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n]}",

    "Human instruction: Put a clean pan in the refrigerator.\nIts scene graph at frame 0 is [pan, sink, fridge]\nOutput: {\n'visual_state_description': 'A dirty pan rests in the sink next to a faucet; the fridge is across the room.',\n'reasoning_and_reflection': 'I will wash the pan, then transfer it into the refrigerator. Opening/closing the fridge door is required.',\n'language_plan': 'Locate a pan and pick it up, find a sink, and put the pan down in the sink. Find a faucet, turn it on to clean the pan, and then turn it off. Pick up the cleaned pan, find the refrigerator, open the refrigerator, place the pan inside, and close the refrigerator.',\n'executable_plan': [\n{\"action_id\": 7, \"action_name\": \"find a Pan\"}\n{\"action_id\": 86, \"action_name\": \"pick up the Pan\"}\n{\"action_id\": 79, \"action_name\": \"find a Sink\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 2, \"action_name\": \"find a Faucet\"}\n{\"action_id\": 155, \"action_name\": \"turn on the Faucet\"}\n{\"action_id\": 156, \"action_name\": \"turn off the Faucet\"}\n{\"action_id\": 7, \"action_name\": \"find a Pan\"}\n{\"action_id\": 86, \"action_name\": \"pick up the Pan\"}\n{\"action_id\": 78, \"action_name\": \"find a Fridge\"}\n{\"action_id\": 139, \"action_name\": \"open the Fridge\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 140, \"action_name\": \"close the Fridge\"}\n]}",

    "Human instruction: Put a box from the couch with a phone in it on the black desk.\nIts scene graph at frame 0 is [cell phone, Box, Desk - black]\nOutput: {\n'visual_state_description': 'A cellphone lies beside an open cardboard box on the couch; a black desk stands opposite.',\n'reasoning_and_reflection': 'I must nest the phone inside the box, then transport the box to the black desk. All are visible in the frame so i will first go to the phone.',\n'language_plan': 'First, find the phone, pick it up, locate the box on the couch, place the phone inside the box, pick up the box, find the black desk, and place the box on the desk.',\n'executable_plan': [\n{\"action_id\": 12, \"action_name\": \"find a CellPhone\"}\n{\"action_id\": 93, \"action_name\": \"pick up the CellPhone\"}\n{\"action_id\": 61, \"action_name\": \"find a Box\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n{\"action_id\": 103, \"action_name\": \"pick up the Box\"}\n{\"action_id\": 49, \"action_name\": \"find a Desk\"}\n{\"action_id\": 133, \"action_name\": \"put down the object in hand\"}\n]}"
]


#common sense, base, and ...
target_examples = [
'''
   Environmental Feedback: The last action succeeded. 
    
   Human Instruction:  Place two yellow bottles on top of a toilet
   
   Last Visual Description: I am in a bathroom. I can see a toilet in front of me. I need to find two yellow bottles and place them on the toilet.
      
   Reasoning and Reflection: The robot was holding the soap bottle and tried to pick up the spray bottle, which is an invalid action. The robot needs to put down the soap bottle first, then pick up the spray bottle, and finally put down the spray bottle on the toilet.
   
   Target Entities in JSON: {"target_names": ['toilet', 'bottle', 'soap bottle', 'spray bottle', 'sink'], "target_attributes": ["yellow"], "target_relations": [["bottle", "on top", "toilet"]], "related_objects": ['toilet', 'bottle', 'soap bottle', 'spray bottle', 'sink', 'bathtub', 'toilet paper', 'soap bar']
}
''',

''' 
    Environmental Feedback: Last action is invalid. Ladle is not visible because it is in DiningTable. Go there to pick the object instead.

    Human Instruction: Rinse off a ladle and move it to the table.
    
    Last Visual Description: I am in a room with a garbage can and a fridge. The floor is tiled. There is a door in the background.

    Reasoning and Reflection: The previous plan failed because I dropped the ladle in the sink and then tried to pick it up without first finding the sink. I need to find the sink again before I can pick up the ladle. After picking up the ladle, I need to turn on the faucet to rinse it. Then, I need to turn off the faucet and find the dining table to put the ladle on it.

    Target Entities in JSON: {
    "target_names": ["ladle", "sink", "faucet", "table"],
    "target_attributes": [],
    "target_relations": [],
    "related_objects": [
        "garbage can", "fridge", "ladle", "sink", "table", "cup", "plate", "chair"
    ]
    }
''', 

''' 
    Environmental Feedback: Last action executed successfully.

    Human Instruction: Rinse off something for serving soup and move it to the table.
    
    Last Visual Description: I can see a bread, an egg, a ladle and a table.

    Reasoning and Reflection: "The instruction is to rinse off something for serving soup and move it to the table. I need to find a suitable object for serving soup, such as a bowl or a ladle. Then, I need to find a faucet to rinse it off. Finally, I need to find a dining table and put the rinsed object on it. I will start by finding a bowl.

    Target Entities in JSON: {
    "target_name": ["soup", "bowl", "ladle", "faucet", "sink", "table", "table"],
    "target_attribute": [],
    "target_relation": [],
    "related_objects": [
        "ladle", "table", "bread", "egg", "sink", "cup", "plate", "chair", "counter top"
    ]
''', 
'''
   Environmental Feedback: Last action is invalid. Robot is currently holding Spoon.
    
   Human Instruction: Set plate with a spoon in it on the kitchen table.
   
   Last Visual Description: I am currently holding a plate. There is a spoon and an apple on the counter. There is a pan on the sink.
   
   Reasoning and Reflection: I have already picked up the plate. Now I need to put down the plate first, then pick up the spoon, and then put the spoon on the plate. Finally, I will put the plate with the spoon on the kitchen table.
   
   Target Entities in JSON: {
       "target_names": ["plate", "spoon", "kitchen table", "counter", "sink", "pan", "apple"],
       "target_attributes": [],
       "target_relations": [["spoon", "in", "plate"], ["plate", "on", "kitchen table"]],
       "related_objects": ["plate", "spoon", "kitchen table", "counter", "sink", "pan", "apple"]
   }
''',

'''
   Environmental Feedback: Last action executed successfully.
    
   Human Instruction: As you prepare for dinner, arrange an eating surface that will hold your food with a spoon in it on the kitchen table.
   
   Last Visual Description: The coffee machine and pan are on the countertop. A knife is also visible. The plate is in the robot's hand.
   
   Reasoning and Reflection: The previous attempt to pick up the spoon failed because the plate was already in hand. The plate needs to be placed on the dining table first before picking up the spoon.
   
   Target Entities in JSON: {
       "target_names": ["plate", "spoon", "kitchen table", "countertop", "coffee machine", "pan", "knife"],
       "target_attributes": [],
       "target_relations": [["spoon", "in", "plate"], ["plate", "on", "kitchen table"]],
       "related_objects": ["plate", "spoon", "kitchen table", "countertop", "coffee machine", "pan", "knife"]
   }
''',

'''
   Environmental Feedback: Last action is invalid. Robot is currently holding SoapBottle.
    
   Human Instruction: Get two silver bottles with brown pumps, put them in dresser drawer.
   
   Last Visual Description: I see two silver bottles with brown pumps on a white countertop. One is in my hand, and the other is on the countertop.
   
   Reasoning and Reflection: I was trying to pick up the second soap bottle while still holding the first one, which is why the action failed. I need to put down the first bottle before picking up the second one. I will find the dresser, open the drawer, put down the bottle in hand, then find the other bottle, pick it up, and put it in the drawer, and finally close the drawer.
   
   Target Entities in JSON: {
       "target_names": ["bottle", "dresser", "dresser drawer", "countertop"],
       "target_attributes": ["silver", "brown pump"],
       "target_relations": [["bottle", "in", "dresser drawer"]],
       "related_objects": ["bottle", "dresser", "dresser drawer", "countertop", "soap bottle"]
   }
''',

'''
   Environmental Feedback: Last action is invalid. Toiletpaper is not visible because it is in GarbageCan. Go there to pick the object instead.
    
   Human Instruction: Put two rolls of toilet paper on the back of a toilet.
   
   Last Visual Description: The robot is currently near the sink. One roll of toilet paper has been placed on the back of the toilet. Another roll of toilet paper is in the garbage can.
   
   Reasoning and Reflection: The previous attempts to pick up the second roll of toilet paper failed because the robot was not in the correct location. The environment feedback indicated that the toilet paper was in the garbage can. This plan corrects that by first finding the toilet paper, then picking it up, then finding the toilet, and finally putting the toilet paper down.
   
   Target Entities in JSON: {
       "target_names": ["toilet paper roll", "toilet", "garbage can", "sink"],
       "target_attributes": ["roll"],
       "target_relations": [["toilet paper roll", "on back", "toilet"], ["toilet paper roll", "in", "garbage can"]],
       "related_objects": ["toilet paper roll", "toilet", "garbage can", "sink"]
   }
''',

'''
   Environmental Feedback: Last action executed successfully.
    
   Human Instruction: Position a soft absorbent fabric in a tub.
   
   Last Visual Description: I am looking at a wall. I need to find a cloth and a bathtub.
   
   Reasoning and Reflection: The user wants me to place a soft absorbent fabric in a tub. First, I need to find the cloth and pick it up. Then, I need to find the bathtub and put the cloth in it.
   
   Target Entities in JSON: {
       "target_names": ["cloth", "bathtub"],
       "target_attributes": ["soft", "absorbent"],
       "target_relations": [["cloth", "in", "bathtub"]],
       "related_objects": ["cloth", "bathtub", "wall"]
   }
''',

'''
   Environmental Feedback: Last action executed successfully.
    
   Human Instruction: To cool a bread slice and place it in the microwave.
   
   Last Visual Description: I am near the microwave. The microwave is closed. I am not holding anything. The bread has been sliced in the previous step.
   
   Reasoning and Reflection: The previous action failed because I was not holding anything. I need to pick up the sliced bread first, then open the microwave, put the bread inside, and close the microwave.
   
   Target Entities in JSON: {
       "target_names": ["bread slice", "microwave"],
       "target_attributes": [],
       "target_relations": [["bread slice", "in", "microwave"]],
       "related_objects": ["bread slice", "microwave"]
   }
''',

'''
   Environmental Feedback: Last action executed successfully.
    
   Human Instruction: Place a spray bottle in the bottom left bathroom cabinet.
   
   Last Visual Description: I can see a white dresser. There is a green spray bottle on top of the dresser. I need to place the spray bottle in the bottom left bathroom cabinet. I assume the dresser is the bathroom cabinet.
   
   Reasoning and Reflection: The task requires me to place a spray bottle in the bottom left bathroom cabinet. First, I need to find the spray bottle and pick it up. Then, I need to find the cabinet, open it, put the spray bottle inside, and close the cabinet.
   
   Target Entities in JSON: {
       "target_names": ["spray bottle", "bathroom cabinet", "dresser"],
       "target_attributes": ["green", "bottom left"],
       "target_relations": [["spray bottle", "in", "bathroom cabinet"]],
       "related_objects": ["spray bottle", "bathroom cabinet", "dresser"]
   }
'''
]

    
exploration_example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_long_horizon_examples.json')

system_prompt = alfred_laser_system_prompt

def bbox_to_mask(bbox, frame_height, frame_width):
    """
    Converts a bounding box into a binary mask using torch tensors.

    Parameters:
    - bbox: A list or tuple of bounding box coordinates [xmin, ymin, xmax, ymax].
    - frame_height: The height of the frame.
    - frame_width: The width of the frame.

    Returns:
    - mask: A binary mask of shape (frame_height, frame_width) with dtype torch.bool.
    """
    # Create an empty mask with the same dimensions as the frame
    mask = np.zeros((frame_height, frame_width, 1), dtype=bool)
    
    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox
    
    # Ensure coordinates are integers and within frame boundaries
    xmin = int(max(0, xmin))
    ymin = int(max(0, ymin))
    xmax = int(min(frame_width - 1, xmax))
    ymax = int(min(frame_height - 1, ymax))
    
    # Set the pixels inside the bounding box to True
    mask[ymin:ymax+1, xmin:xmax+1, :] = True
    
    return mask

class EB_AlfredEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None
        # self.gd_checkpoint_path = config['gd_checkpoint_path']

        # self.gd_config_path = config['gd_config_path']
        
        self.gd_checkpoint_path= "/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth"
        self.gd_config_path = "/home/asethi04/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

    def check_config_valid(self):
        if self.config['multistep'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multistep, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multistep']:
                logger.warning("Language only mode should not have multistep enabled. Setting these arguments to False ...")
                self.config['multistep'] = 0
        
    def save_episode_metric(self, episode_info):
        episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        filename = 'episode_{}_final_res.json'.format(episode_idx)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        valid_eval_sets = list(valid_eval_sets)
        if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
            valid_eval_sets = ValidEvalSets

        print(valid_eval_sets)
        
        
        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"
            self.env = EBAlfEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], 
                                          exp_name=exp_name, selected_indexes=self.config.get('selected_indexes', []), 
                                          detection_box=False,
                                          resolution=self.config.get('resolution', 500), 
                                          )
            examples = json.load(open(example_path, 'r+')) if self.eval_set != 'long_horizon' else json.load(open(exploration_example_path, 'r+'))
            
            if self.config['sg_text']:
                few_shot_examples = laser_sg_examples_no_revert
            else:
                few_shot_examples = examples
            model_type = self.config.get('model_type', 'remote')
            self.planner = VLMPlanner(self.model_name, model_type, self.env.language_skill_set, system_prompt, examples=few_shot_examples, scene_graph_examples=target_examples, n_shot=self.config['n_shots'], 
                                            obs_key='head_rgb', chat_history=self.config['chat_history'], language_only=self.config['language_only'],
                                            use_feedback=self.config.get('env_feedback', True), multistep=self.config.get('multistep', 0), 
                                            boundingbox=self.config['detection_box'], scene_graph_text=self.config['sg_text'],
                                            gd_only=self.config['gd_only'], tp=self.config['tp'],
                                            blockingobj=self.config['block_objects'],
                                            gd_checkpoint_path=self.gd_checkpoint_path, gd_config_path=self.gd_config_path,
                                            top_k=self.config['top_k'], aggr_thres=self.config['aggr_thres'])

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), output_file='summary.json')
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            
            
            current_frames = [obs['head_rgb']]
            
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            # update the action space for alfred due to dynamic objects
            self.planner.set_actions(self.env.language_skill_set)
            done = False
            while not done:
                try: 
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")
                    if action == -2: # empty plan stop here
                        episode_info['empty_plan'] = 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -2,
                            'action_description': 'empty plan',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        break 
                    if action == -1:
                        self.env._cur_invalid_actions += 1
                        episode_info['reward'].append(-1)
                        episode_info['num_invalid_actions'] += 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'invalid action',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                            break
                        continue
                    
                    # mutiple actions
                    if type(action) == list:
                        current_frames = []
                        for action_single in action[:min(self.env._max_episode_steps - self.env._current_step, len(action))]:
                            obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
                            action_str = action_single if type(action_single) == str else self.env.language_skill_set[action_single]
                            current_frames.append(obs['head_rgb'])
                            print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                            print(f"Last action success: {info['last_action_success']}")
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                            episode_info['reward'].append(reward)
                            episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                            if done or not info['last_action_success']:
                                # stop or replanning
                                print("Invalid action or task complete. If invalid then Replanning.")
                                break
                    else: # single action
                        obs, reward, done, info = self.env.step(action, reasoning=reasoning)
                        action_str = action if type(action) == str else self.env.language_skill_set[action]
                        current_frames = [obs['head_rgb']]
                        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)
                        episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                
                except Exception as e: 
                    print(e)
                    time.sleep(30)

            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info["task_progress"] = info['task_progress']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["num_invalid_actions"] = episode_info['num_invalid_actions']
            episode_info["num_invalid_action_ratio"] = episode_info['num_invalid_actions'] / info["env_step"] if info['env_step'] > 0 else 0
            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)

            self.env.save_episode_log()
            self.save_episode_metric(episode_info)
            progress_bar.update()


if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Change configuration parameters.')
        parser.add_argument('--model_name', default='gemini-2.0-flash', type=str, help='Name of the model.')
        parser.add_argument('--n_shots', type=int, help='Number of examples')
        parser.add_argument('--down_sample_ratio', type=float, help='Down sample ratio.')
        parser.add_argument('--model_type', default='remote', type=str, help='Type of the model.')
        parser.add_argument('--language_only', type=int, help='Set to True for language only mode.')
        parser.add_argument('--exp_name', default='laser_alfred_new_prompt_gemini_gd_boxes_with_labels', type=str, help='Name of the experiment.')
        parser.add_argument('--chat_history', type=int, help='Set to True to enable chat history.')
        parser.add_argument('--detection_box', type=int, help='Set to True to enable detection.')
        parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
        parser.add_argument('--multistep', type=int, help='Number of steps for multi-step reasoning.')
        parser.add_argument('--resolution', default=500, type=int, help='Resolution for processing.')
        parser.add_argument('--env_feedback', type=int, help='Set to True to enable environment feedback.')
        parser.add_argument('--tp', default=4, type=int, help='number of tensor parallel splits of the model parameters')
        return parser.parse_args()


    config = {
        'model_name': 'gemini-2.0-flash',  # 'Qwen/Qwen2-VL-7B-Instruct',
        'n_shots': 10,
        'down_sample_ratio': 1.0,
        'model_type': 'remote', # 'local', 
        'language_only': 0,
        # 'exp_name': 'laser_alfred_old_few_shot_gemini',
        'chat_history': 0, 
        'detection_box': True,
        'multiview': False, 
        # 'eval_sets': ['base', 'spatial', 'visual_appearance', 'common_sense', 'complex_instruction', 'long_horizon'], 
        'eval_sets': ['base'],
        'selected_indexes': [], 
        'multistep':0, 
        'block_objects': False,
        'sg_text': True,
        'resolution': 500, 
        'env_feedback': 1,
        'tp': 4,
        'action_num_per_plan': 5,
        'fov': 100,
        'sleep_time': 0,
        'purpose': "retest",
        'icl_abl':0, 
        'visual':0,
        'gd_only': False,
        'top_k': 5, 
        'aggr_thres': 0.8,
        'gd_config_path': '/home/asethi04/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        'gd_checkpoint_path': '/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth'
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    evaluator = EB_AlfredEvaluator(config)
    evaluator.evaluate_main()




