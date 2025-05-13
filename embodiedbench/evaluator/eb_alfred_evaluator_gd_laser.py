import os

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

laser_sg_examples_no_revert = [

'''
Human instruction: Pick up the alarm clock and turn on the lamp.
The target state: I am holding an alarm clock and there should be a lightened lamp in my sight.
The current state: I am in a bedroom.  A small alarm clock sits on the bedside table next to a switched-off desk lamp. 
The current scene graph: 
    The objects are: [alarm_clock_1, bedside_table_1, lamp_1]
    The attributes are: []
    The relations are: [(alarm_clock_1, on, bedside_table_1), (alarm_clock_1, next to, lamp_1)]
The target state:
    The objects are: [alarm_clock_1, lamp_1]
    The attributes are: [(holding, alarm_clock_1), (lightened, lamp_1)]
    The relations are: []

Output: {
    'visual_state_description': 'I am in a bedroom. A small alarm clock sits on the bedside table next to a switched-off desk lamp.',
    'reasoning_and_reflection': 
        'I need to check each part in the target state whether they are statisied.
         I should hold an alarm clock. I can see an alarm clock in the scene. I am not holding an alarm clock now. Not satisfied.
         I should lighten the lamp.  I can see a lamp in the scene. It is not lightened. Not satisfied.
         I must hold the alarm clock and then illuminate the area by turning on the desk lamp.',
    'language_plan': 'First, locate the alarm clock in the room. Once found, carefully pick up the alarm clock. Next, identify the location of the desk lamp. After locating the desk lamp, proceed to turn it on.',
    'executable_plan': [\n{\"action_id\": 57, \"action_name\": \"find a AlarmClock\"}\n{\"action_id\": 98, \"action_name\": \"pick up the AlarmClock\"}\n{\"action_id\": 63, \"action_name\": \"find a DeskLamp\"}\n{\"action_id\": 151, \"action_name\": \"turn on the DeskLamp\"}
    ]}
''',

'''
Human instruction: Examine a credit card by the light of a tall lamp.
The target state: I am holding a credit card, and a tall lamp is lightened in the scene so that I can see the card clearly.
The current state: I am in a lounge with a floor lamp in the corner and a credit card lying on a coffee table.
The current scene graph: 
    The objects are: [credit_card_1, coffee_table_1, floor_lamp_1]
    The attributes are: []
    The relations are: [(credit_card_1, on, coffee_table_1)]
The target state:
    The objects are: [credit_card_1, floor_lamp_1]
    The attributes are: [(holding, credit_card_1), (lightened, floor_lamp_1)]
    The relations are: []
    
Output: {
'visual_state_description': 'A lounge with a floor lamp in the corner and a credit card lying on a coffee table.',
'reasoning_and_reflection': 'I need to assess whether the current scene satisfies the target state. 
    First, I should be holding a credit card. 
    Currently, the credit card is on the coffee table, not in my possession, so this condition is not satisfied. 
    Second, the floor lamp should be lightened to help me examine the credit card. 
    At present, the floor lamp is in the corner but not turned on, so this condition is also not satisfied. 
    Therefore, to achieve the target state, I must pick up the credit card from the coffee table and turn on the floor lamp to illuminate the area.'
'language_plan': 'Start by locating the credit card in the room. Once you have found the credit card, proceed to pick it up. Then, search for the floor lamp and find its location. After finding the floor lamp, turn it on to illuminate the area so you can examine the credit card effectively.',
'executable_plan': [
    {"action_id": 56, "action_name": "find a CreditCard"}
    {"action_id": 96, "action_name": "pick up the CreditCard"}
    {"action_id": 27, "action_name": "find a FloorLamp"}
    {"action_id": 153, "action_name": "turn on the FloorLamp"}
]}
''',

'''
Human instruction: Use lamp light to look at the remote.
The target state: I am holding a remote control, and a floor lamp is turned on so I can inspect the remote clearly.
The current state: I am in a living room. A sofa faces a dark floor lamp. A remote control rests on the arm of the sofa.
The current scene graph: 
    The objects are: [remote_control_1, sofa_1, floor_lamp_1]
    The attributes are: []
    The relations are: [(remote_control_1, on, arm_of_sofa_1), (sofa_1, facing, floor_lamp_1)]
The target state:
    The objects are: [remote_control_1, floor_lamp_1]
    The attributes are: [(holding, remote_control_1), (lightened, floor_lamp_1)]
    The relations are: []
    
Output: {
'visual_state_description': 'A sofa faces a dark floor lamp. A remote control rests on the arm of the sofa.',
'reasoning_and_reflection': 'I need to determine if the current state fulfills the goal. 
    First, I should be holding the remote control. However, it is currently resting on the arm of the sofa, so this is not satisfied. 
    Second, the floor lamp should be lightened to provide visibility. 
    At the moment, the floor lamp is off, which also does not meet the target condition. 
    Therefore, I need to pick up the remote control and turn on the floor lamp to achieve the desired state.',
'language_plan': 'First, locate the remote control in the room. Once the remote control is found, pick it up carefully. Next, search for the floor lamp and determine its location. After you have located the floor lamp, turn it on to provide sufficient light for examining the remote control.',
'executable_plan': [
{"action_id": 68, "action_name": "find a RemoteControl"}
{"action_id": 116, "action_name": "pick up the RemoteControl"}
{"action_id": 27, "action_name": "find a FloorLamp"}
{"action_id": 153, "action_name": "turn on the FloorLamp"}
]}
''',

'''
Human instruction: Set the box on the table.
The target state: The box should be placed on top of the dining table.
The current state: I am in a dining area. A cardboard box is on the floor near a dining table.
The current scene graph: 
    The objects are: [box_1, dining_table_1]
    The attributes are: []
    The relations are: [(box_1, near, dining_table_1), (box_1, on, floor)]
The target state:
    The objects are: [box_1, dining_table_1]
    The attributes are: []
    The relations are: [(box_1, on, dining_table_1)]
    
Output: {
'visual_state_description': 'A cardboard box is on the floor near a dining table.',
'reasoning_and_reflection': 'I need to check if the current configuration satisfies the target state. 
    The goal is for the box to be on the dining table. Right now, the box is on the floor near the table, which does not fulfill the required spatial relation. 
    Therefore, I need to pick up the box and place it directly on top of the dining table to achieve the target state.',
'language_plan': 'First, locate the box within the room. After finding the box, gently pick it up. Next, search for the dining table and identify its location. Once the dining table is found, place the box on the table carefully.',
'executable_plan': [
{"action_id": 61, "action_name": "find a Box"}
{"action_id": 103, "action_name": "pick up the Box"}
{"action_id": 18, "action_name": "find a DiningTable"}
{"action_id": 133, "action_name": "put down the object in hand"}
]}
''',

'''
Human instruction: Place a heated slice of tomato in the kitchen sink.
The target state: A slice of tomato that has been heated is placed inside the kitchen sink.
The current state: I am in a kitchen. A whole tomato sits on the counter; a microwave and a sink are visible nearby.
The current scene graph: 
    The objects are: [tomato_1, counter_1, microwave_1, sink_1]
    The attributes are: []
    The relations are: [(tomato_1, on, counter_1)]
The target state:
    The objects are: [tomato_slice_1, sink_1]
    The attributes are: [(heated, tomato_slice_1)]
    The relations are: [(tomato_slice_1, in, sink_1)]
    
Output: {
'visual_state_description': 'A whole tomato sits on the counter; a microwave and sink are visible nearby.',
'reasoning_and_reflection': 'To satisfy the target state, I need a heated slice of tomato to be placed inside the sink. 
    Currently, there is only a whole tomato on the counter, and the tomato is not sliced or heated. The target state is not satisfied.
    Thus, I need to slice the tomato, heat the slice in the microwave, and place the heated slice into the sink.',
'language_plan': 'Find a butter knife and pick it up, locate a tomato and slice it. Find a sink and put down the knife. Locate a slice of tomato and pick it up, find a microwave, and open it. Place the tomato slice inside and close the microwave. Turn on the microwave to heat the tomato slice, then turn off the microwave and open it. Pick up the heated tomato slice, close the microwave, locate the kitchen sink, and place the heated tomato slice in the sink.',
'executable_plan': [
{"action_id": 36, "action_name": "find a ButterKnife"}
{"action_id": 130, "action_name": "pick up the ButterKnife"}
{"action_id": 35, "action_name": "find a Tomato"}
{"action_id": 159, "action_name": "slice the Tomato"}
{"action_id": 79, "action_name": "find a Sink"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 35, "action_name": "find a Tomato"}
{"action_id": 129, "action_name": "pick up the Tomato"}
{"action_id": 38, "action_name": "find a Microwave"}
{"action_id": 143, "action_name": "open the Microwave"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 144, "action_name": "close the Microwave"}
{"action_id": 149, "action_name": "turn on the Microwave"}
{"action_id": 150, "action_name": "turn off the Microwave"}
{"action_id": 143, "action_name": "open the Microwave"}
{"action_id": 35, "action_name": "find a Tomato"}
{"action_id": 129, "action_name": "pick up the Tomato"}
{"action_id": 144, "action_name": "close the Microwave"}
{"action_id": 79, "action_name": "find a Sink"}
{"action_id": 133, "action_name": "put down the object in hand"}
]}
''',

'''
Human instruction: Move the towel on the hanger into the cabinet.
The target state: The hand towel should be placed inside the cabinet, and the cabinet should be closed.
The current state: I am in a bathroom. A hand towel hangs from a wall-mounted holder; a closed cabinet stands nearby.
The current scene graph: 
    The objects are: [hand_towel_1, hanger_1, cabinet_1]
    The attributes are: [(closed, cabinet_1)]
    The relations are: [(hand_towel_1, on, hanger_1), (cabinet_1, near, hanger_1)]
The target state:
    The objects are: [hand_towel_1, cabinet_1]
    The attributes are: [(in, hand_towel_1, cabinet_1), (closed, cabinet_1)]
    The relations are: []
    
Output: {
'visual_state_description': 'A hand towel hangs from a wall-mounted holder; a closed cabinet stands nearby.',
'reasoning_and_reflection': 
    'The hand towel is currently on the hanger and not inside the cabinet. Not satisfied. 
     The cabinet is closed, so I must open it before placing the towel inside, and close it again afterward to satisfy the target state. 
     Therefore, I need to retrieve the towel, open the cabinet, store the towel, and then close the cabinet.',
'language_plan': 'Begin by locating the hand towel on the hanger. Once you have found the hand towel, proceed to pick it up. Next, search for the cabinet and identify its location. After finding the cabinet, open it. Place the hand towel inside the cabinet, and finally, close the cabinet door.',
'executable_plan': [
{"action_id": 9, "action_name": "find a HandTowel"}
{"action_id": 89, "action_name": "pick up the HandTowel"}
{"action_id": 48, "action_name": "find a Cabinet"}
{"action_id": 145, "action_name": "open the Cabinet"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 146, "action_name": "close the Cabinet"}
]}''',

'''
Human instruction: Place the microwaved slice of apple on to the white table.
The target state: A microwaved apple slice is placed on the white dining table.
The current state: I am in a kitchen. A whole apple lies next to a microwave; a white dining table is visible across the kitchen.
The current scene graph: 
    The objects are: [apple_1, microwave_1, dining_table_1]
    The attributes are: [(color, dining_table_1, white), (closed, microwave_1)]
    The relations are: [(apple_1, next to, microwave_1), (dining_table_1, across_from, microwave_1)]
The target state:
    The objects are: [microwaved_apple_slice_1, dining_table_1]
    The attributes are: [(heated, microwaved_apple_slice_1)]
    The relations are: [(microwaved_apple_slice_1, on, dining_table_1)]
    
Output: {
'visual_state_description': 'A whole apple lies next to a microwave; a white dining table is visible across the kitchen.',
'reasoning_and_reflection': 'I need to check each part in the target state to see whether it is satisfied. 
I should have a microwaved slice of apple placed on the white dining table.
I can see a whole apple in the scene, but it has not been sliced. Not satisfied. 
I should heat the slice using the microwave. The microwave is visible, but no slice is currently inside or being heated. Not satisfied. 
I should then place the heated slice on the white dining table. The table is visible, but it is currently empty. Not satisfied. 
Therefore, I must slice the apple, microwave a slice, and place the microwaved slice on the white dining table.'
'language_plan': 'Locate an apple and pick it up, find a dining table and put the apple down, look for a knife and pick it up, find another apple and slice it, find the dining table and put the knife down, locate an apple and pick it up, search for a microwave and open it, put the apple in the microwave and close it, turn on the microwave to heat the apple slice, turn off the microwave and open it, pick up the microwaved apple slice, close the microwave, find the white table, and place the microwaved apple slice on the white table.',
'executable_plan': [
{"action_id": 22, "action_name": "find a Apple"}
{"action_id": 107, "action_name": "pick up the Apple"}
{"action_id": 18, "action_name": "find a DiningTable"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 34, "action_name": "find a Knife"}
{"action_id": 127, "action_name": "pick up the Knife"}
{"action_id": 22, "action_name": "find a Apple"}
{"action_id": 160, "action_name": "slice the Apple"}
{"action_id": 18, "action_name": "find a DiningTable"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 22, "action_name": "find a Apple"}
{"action_id": 107, "action_name": "pick up the Apple"}
{"action_id": 38, "action_name": "find a Microwave"}
{"action_id": 143, "action_name": "open the Microwave"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 144, "action_name": "close the Microwave"}
{"action_id": 149, "action_name": "turn on the Microwave"}
{"action_id": 150, "action_name": "turn off the Microwave"}
{"action_id": 143, "action_name": "open the Microwave"}
{"action_id": 22, "action_name": "find a Apple"}
{"action_id": 107, "action_name": "pick up the Apple"}
{"action_id": 144, "action_name": "close the Microwave"}
{"action_id": 18, "action_name": "find a DiningTable"}
{"action_id": 133, "action_name": "put down the object in hand"}
]}
''',

'''
Human instruction: slice a loaf of bread put a slice on the counter.
The target state: A sliced piece of bread is placed on the countertop. The knife is no longer in use.
The current state: I am in a kitchen. A loaf of bread sits beside a bread knife on the countertop.
The current scene graph: 
    The objects are: [bread_loaf_1, bread_knife_1, countertop_1]
    The attributes are: []
    The relations are: [(bread_loaf_1, beside, bread_knife_1), (bread_loaf_1, on, countertop_1), (bread_knife_1, on, countertop_1)]
The target state:
    The objects are: [bread_slice_1, countertop_1]
    The attributes are: []
    The relations are: [(bread_slice_1, on, countertop_1)]
    
Output: {
'visual_state_description': 'A loaf of bread sits beside a bread knife on the countertop.',
'reasoning_and_reflection': 'I need to check each part in the target state to see whether it is satisfied. 
I should have a sliced piece of bread placed on the countertop. I can see a loaf of bread and a knife in the scene, but the bread has not been sliced. Not satisfied.
I can also see the countertop, but there is no slice of bread on it. Not satisfied. 
Therefore, I must use the knife to slice the bread and place one of the slices on the countertop.',
'language_plan': 'First locate a knife and pick it up, find a loaf of bread, and slice the bread. Find a cabinet, open it, and put the knife inside, then close the cabinet. Find a slice of bread, pick it up, find the countertop, and place the slice of bread on the counter.',
'executable_plan': [
{"action_id": 34, "action_name": "find a Knife"}
{"action_id": 127, "action_name": "pick up the Knife"}
{"action_id": 53, "action_name": "find a Bread"}
{"action_id": 161, "action_name": "slice the Bread"}
{"action_id": 48, "action_name": "find a Cabinet"}
{"action_id": 145, "action_name": "open the Cabinet"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 146, "action_name": "close the Cabinet"}
{"action_id": 53, "action_name": "find a Bread"}
{"action_id": 92, "action_name": "pick up the Bread"}
{"action_id": 78, "action_name": "find a Fridge"}
{"action_id": 139, "action_name": "open the Fridge"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 140, "action_name": "close the Fridge"}
{"action_id": 139, "action_name": "open the Fridge"}
{"action_id": 53, "action_name": "find a Bread"}
{"action_id": 92, "action_name": "pick up the Bread"}
{"action_id": 140, "action_name": "close the Fridge"}
{"action_id": 39, "action_name": "find a CounterTop"}
{"action_id": 133, "action_name": "put down the object in hand"}
]}
''',
'''
Human instruction: Put a clean pan in the refrigerator.
The target state: A clean pan is placed inside the refrigerator, and the fridge is closed.
The current state: I am in a kitchen. A dirty pan rests in the sink next to a faucet; the fridge is across the room.
The current scene graph: 
    The objects are: [pan_1, sink_1, faucet_1, fridge_1]
    The attributes are: [(dirty, pan_1), (closed, fridge_1)]
    The relations are: [(pan_1, in, sink_1), (sink_1, next to, faucet_1), (fridge_1, across_from, sink_1)]
The target state:
    The objects are: [pan_1, fridge_1]
    The attributes are: [(clean, pan_1), (closed, fridge_1)]
    The relations are: [(pan_1, in, fridge_1)]
    
Output: {
'visual_state_description': 'A dirty pan rests in the sink next to a faucet; the fridge is across the room.',
'reasoning_and_reflection': 'I need to check each part in the target state to see whether it is satisfied. 
I should have a clean pan placed inside the refrigerator. I can see a pan, but it is currently dirty and sitting in the sink. Not satisfied. 
The fridge is visible but closed, and the pan is not inside it. Not satisfied. 
Therefore, I must wash the pan to make it clean, then place it into the refrigerator and close the door.',
'language_plan': 'Locate a pan and pick it up, find a sink, and put the pan down in the sink. Find a faucet, turn it on to clean the pan, and then turn it off. Pick up the cleaned pan, find the refrigerator, open the refrigerator, place the pan inside, and close the refrigerator.',
'executable_plan': [
{"action_id": 7, "action_name": "find a Pan"}
{"action_id": 86, "action_name": "pick up the Pan"}
{"action_id": 79, "action_name": "find a Sink"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 2, "action_name": "find a Faucet"}
{"action_id": 155, "action_name": "turn on the Faucet"}
{"action_id": 156, "action_name": "turn off the Faucet"}
{"action_id": 7, "action_name": "find a Pan"}
{"action_id": 86, "action_name": "pick up the Pan"}
{"action_id": 78, "action_name": "find a Fridge"}
{"action_id": 139, "action_name": "open the Fridge"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 140, "action_name": "close the Fridge"}
]}''',

'''
Human instruction: Put a box from the couch with a phone in it on the black desk.
The target state: A phone is inside the box, and the box is placed on the black desk.
The current state: I am in a living room. A cellphone lies beside an open cardboard box on the couch; a black desk stands opposite.
The current scene graph: 
    The objects are: [cellphone_1, box_1, couch_1, desk_1]
    The attributes are: [(color, desk_1, black), (open, box_1)]
    The relations are: [(cellphone_1, beside, box_1), (box_1, on, couch_1), (desk_1, opposite, couch_1)]
The target state:
    The objects are: [cellphone_1, box_1, desk_1]
    The attributes are: []
    The relations are: [(cellphone_1, in, box_1), (box_1, on, desk_1)]

Output: {
'visual_state_description': 'A cellphone lies beside an open cardboard box on the couch; a black desk stands opposite.',
'reasoning_and_reflection': 'I need to check each part in the target state to see whether it is satisfied. 
First, the phone should be inside the box. Currently, the phone is next to the box on the couch, not inside it. Not satisfied. 
Second, the box should be on the black desk. Currently, the box is on the couch. Not satisfied. 
Therefore, I must first place the phone inside the box, then move the box from the couch to the black desk.',
'language_plan': 'First, find the phone, pick it up, locate the box on the couch, place the phone inside the box, pick up the box, find the black desk, and place the box on the desk.',
'executable_plan': [
{"action_id": 12, "action_name": "find a CellPhone"}
{"action_id": 93, "action_name": "pick up the CellPhone"}
{"action_id": 61, "action_name": "find a Box"}
{"action_id": 133, "action_name": "put down the object in hand"}
{"action_id": 103, "action_name": "pick up the Box"}
{"action_id": 49, "action_name": "find a Desk"}
{"action_id": 133, "action_name": "put down the object in hand"}
]}
'''
]

#common sense, base, and ...
target_examples = [
'''
   Environmental Feedback of last action: The last action succeed. 
      
   Reasoning and Reflection of last action:  The robot shall put down the spray bottle on the toilet.
   
   Human Instruction: Place two yellow bottles on top of a toilet.
      
   Visual Description: I am in a bathroom. I can see a toilet in front of me, and I am holding one soap bottle. There is another soap bottle on the top of another toilet.

   Target state and current state in JSON: {
        "target_state": "there are two yellow bottles on the top of a toilet."
        "target_objects": ["yellow_bottle_1", "yellow_bottle_2", "toilet_1"],
        "target_attributes": []
        "target_relations": [["yellow_bottle_1", "on", "toilet_1"], ["yellow_bottle_2", "on", "toilet_1"]], 
        "current_state": "there are two pink bottles and a toilet in the scene. One pink bottle is on the top of the toilet. I am holding another pink bottle."
        "current_objects": ["pink_bottle_1", "pink_bottle_2", "toilet_1"],
        "current_attrbutes": []
        "current_relations": [["pink_bottle_1", "on", "toilet_1"], ["pink_bottle_2", "in front of", "toilet_1"]], 
        }
''',

''' 
    Environmental Feedback: Last action is invalid. Ladle is not visible because it is in DiningTable. Go there to pick the object instead.

    Reasoning and Reflection: The previous plan failed because Ladle is not visible. I need to find the dinning table before I can pick up the ladle. After picking up the ladle, I need to turn on the faucet to rinse it. Then, I need to turn off the faucet and find the dining table to put the ladle on it.

    Human Instruction: Rinse off a ladle and move it to the table.
    
    Visual Description: I am in a room with a garbage can and a fridge. The floor is tiled. There is a door in the background.

    Target state and current state in JSON: {
        "target_state": "There is a clean ladle on the top of the table."
        "target_objects": ["ladle_1", "table_1"],
        "target_attributes": []
        "target_relations": [["ladle_1", "on", "table_1"]], 
        "current_state": "There is a garbage can, fridge and door in the scene."
        "current_objects": ["garbage can", "fridge", "door"],
        "current_attrbutes": []
        "current_relations": [], 
        }
''', 

''' 
    Environmental Feedback: Last action executed successfully.
    
    Reasoning and Reflection: "The instruction is to rinse off something for serving soup and move it to the table. I need to find a suitable object for serving soup, such as a bowl or a ladle. Then, I need to find a faucet to rinse it off. Finally, I need to find a dining table and put the rinsed object on it. I will start by finding a bowl or a laddle.

    Human Instruction: Rinse off something for serving soup and move it to the table.
    
    Visual Description: I can see a bread, an egg, a ladle and a table.
    
    Target state and current state in JSON: {
        "target_state": "There is a ladle on a table."
        "target_objects": ["ladle_1", "table_1"],
        "target_attributes": []
        "target_relations": [["ladle_1", "on", "table_1"]], 
        "current_state": "There is a bread,  an egg, a ladle and a table"
        "current_objects": ["ladle_1", "table_1", "egg_1", "bread_1"],
        "current_attrbutes": []
        "current_relations": [["ladle_1", "on", "table_1"], ["egg_1", "on", "table_1"], ["bread_1", "on", "table_1"]], 
        }
''', 

'''
   Environmental Feedback: Last action is invalid. Robot is currently holding Spoon.
   
   Reasoning and Reflection: I have already picked up the plate. Now I need to put down the plate first, then pick up the spoon, and then put the spoon on the plate. Finally, I will put the plate with the spoon on the kitchen table.
 
   Human Instruction: Set plate with a spoon in it on the kitchen table.
   
   Visual Description: I am currently holding a plate. There is a spoon and an apple on the counter. There is a pan on the sink.
   
   Target state and current state in JSON: {
        "target_state": "There is a spoon on a plate. The plate is on a kitchen table."
        "target_objects": ["spoon_1", "plate_1", "kitchen_table_1"],
        "target_attributes": []
        "target_relations": [["spoon_1", "in", "plate_1"], ["plate_1", "on", "kitchen_table_1"]], 
        "current_state": "I am currently holding a plate. There is a spoon and an apple on the counter. There is a pan on the sink."
        "current_objects": ["plate_1", "spoon_1", "apple_1", "counter_1", "sink_1", "pan_1"],
        "current_attrbutes": [["holding", "plate_1"]]
        "current_relations": [["spoon_1", "on", "counter_1"], ["apple_1", "on", "counter_1"], ["pan_1", "in", "sink_1"]], 
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
                                            top_k=self.config['top_k'], aggr_thres=self.config['aggr_thres'], 
                                            gd_box_threshold=self.config['gd_box_threshold'], gd_text_threshold=self.config['gd_text_threshold'])

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
                # try: 
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
                
                # except Exception as e: 
                #     print(e)
                #     time.sleep(30)

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
        parser.add_argument('--exp_name', default='gd_laser', type=str, help='Name of the experiment.')
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
        'top_k': 3, 
        'aggr_thres': 0.4,
        'gd_config_path': '/home/asethi04/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        'gd_checkpoint_path': '/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth',
        'gd_box_threshold': 0.25, 
        'gd_text_threshold': 0.3
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    evaluator = EB_AlfredEvaluator(config)
    evaluator.evaluate_main()




