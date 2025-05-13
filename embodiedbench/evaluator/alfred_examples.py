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
    "visual_state_description": "I am in a bedroom. A small alarm clock sits on the bedside table next to a switched-off desk lamp.",
    "reasoning_and_reflection": "I need to check each part in the target state whether they are statisied.\\n I should hold an alarm clock. I can see an alarm clock in the scene. I am not holding an alarm clock now. Not satisfied.\\n I should lighten the lamp. I can see a lamp in the scene. It is not lightened. Not satisfied.\\n I must hold the alarm clock and then illuminate the area by turning on the desk lamp.\\n I need to check the pre- and post-conditions for the actions.\\n The \"Hold\" action requires nothing in hand. I am not holding any thing in hand, satisfied.",
    "language_plan": "First, locate the alarm clock in the room. Once found, carefully pick up the alarm clock. Next, identify the location of the desk lamp. After locating the desk lamp, proceed to turn it on.",
    "executable_plan": [
        {"action_id": 57, "action_name": "find a AlarmClock"},
        {"action_id": 98, "action_name": "pick up the AlarmClock"},
        {"action_id": 63, "action_name": "find a DeskLamp"},
        {"action_id": 151, "action_name": "turn on the DeskLamp"}
    ]
}
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
    "visual_state_description": "A lounge with a floor lamp in the corner and a credit card lying on a coffee table.",
    "reasoning_and_reflection": "I need to assess whether the current scene satisfies the target state.\\n First, I should be holding a credit card.\\n Currently, the credit card is on the coffee table, not in my possession, so this condition is not satisfied.\\n Second, the floor lamp should be lightened to help me examine the credit card.\\n At present, the floor lamp is in the corner but not turned on, so this condition is also not satisfied.\\n Therefore, to achieve the target state, I must pick up the credit card from the coffee table and turn on the floor lamp to illuminate the area.\\n I also need to be aware of the pre-condition and post condition of the actions I have planed.\\n pick up requires I am holding nothing.\\n Because I am holding nothing, the precondition is satisfied.",
    "language_plan": "Start by locating the credit card in the room. Once you have found the credit card, proceed to pick it up. Then, search for the floor lamp and find its location. After finding the floor lamp, turn it on to illuminate the area so you can examine the credit card effectively.",
    "executable_plan": [
        {"action_id": 56, "action_name": "find a CreditCard"},
        {"action_id": 96, "action_name": "pick up the CreditCard"},
        {"action_id": 27, "action_name": "find a FloorLamp"},
        {"action_id": 153, "action_name": "turn on the FloorLamp"}
    ]
}
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
    "visual_state_description": "A sofa faces a dark floor lamp. A remote control rests on the arm of the sofa.",
    "reasoning_and_reflection": "I need to determine if the current state fulfills the goal.\\n First, I should be holding the remote control. However, it is currently resting on the arm of the sofa, so I will need to use a \"Find\" action to find the remote control.\\n Second, the floor lamp should be lightened to provide visibility.\\n At the moment, the floor lamp is off, which also does not meet the target condition.\\n Therefore, I need to pick up the remote control and turn on the floor lamp to achieve the desired state.",
    "language_plan": "First, locate the remote control in the room. Once the remote control is found, pick it up carefully. Next, search for the floor lamp and determine its location. After you have located the floor lamp, turn it on to provide sufficient light for examining the remote control.",
    "executable_plan": [
        {"action_id": 68, "action_name": "find a RemoteControl"},
        {"action_id": 116, "action_name": "pick up the RemoteControl"},
        {"action_id": 27, "action_name": "find a FloorLamp"},
        {"action_id": 153, "action_name": "turn on the FloorLamp"}
    ]
}
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
    "visual_state_description": "A cardboard box is on the floor near a dining table.",
    "reasoning_and_reflection": "I need to check if the current configuration satisfies the target state.\\n The goal is for the box to be on the dining table. Right now, the box is on the floor near the table, which does not fulfill the required spatial relation.\\n Therefore, I need to pick up the box and place it directly on top of the dining table to achieve the target state.",
    "language_plan": "First, locate the box within the room using the \"Find\" action. After finding the box, gently pick it up. Next, search for the dining table and identify its location. Once the dining table is found, place the box on the table carefully.",
    "executable_plan": [
        {"action_id": 61, "action_name": "find a Box"},
        {"action_id": 103, "action_name": "pick up the Box"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
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
    "visual_state_description": "A whole tomato sits on the counter; a microwave and sink are visible nearby.",
    "reasoning_and_reflection": "To satisfy the target state, I need a heated slice of tomato to be placed inside the sink.\\n Currently, there is only a whole tomato on the counter, and the tomato is not sliced or heated. The target state is not satisfied.\\n Thus, I need to slice the tomato, heat the slice in the microwave, and place the heated slice into the sink.",
    "language_plan": "Find a butter knife and pick it up, locate a tomato and slice it. Find a sink and put down the knife. Locate a slice of tomato and pick it up, find a microwave, and open it. Place the tomato slice inside and close the microwave. Turn on the microwave to heat the tomato slice, then turn off the microwave and open it. Pick up the heated tomato slice, close the microwave, locate the kitchen sink, and place the heated tomato slice in the sink.",
    "executable_plan": [
        {"action_id": 36, "action_name": "find a ButterKnife"},
        {"action_id": 130, "action_name": "pick up the ButterKnife"},
        {"action_id": 35, "action_name": "find a Tomato"},
        {"action_id": 159, "action_name": "slice the Tomato"},
        {"action_id": 79, "action_name": "find a Sink"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 35, "action_name": "find a Tomato"},
        {"action_id": 129, "action_name": "pick up the Tomato"},
        {"action_id": 38, "action_name": "find a Microwave"},
        {"action_id": 143, "action_name": "open the Microwave"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 144, "action_name": "close the Microwave"},
        {"action_id": 149, "action_name": "turn on the Microwave"},
        {"action_id": 150, "action_name": "turn off the Microwave"},
        {"action_id": 143, "action_name": "open the Microwave"},
        {"action_id": 35, "action_name": "find a Tomato"},
        {"action_id": 129, "action_name": "pick up the Tomato"},
        {"action_id": 144, "action_name": "close the Microwave"},
        {"action_id": 79, "action_name": "find a Sink"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
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
    "visual_state_description": "A hand towel hangs from a wall-mounted holder; a closed cabinet stands nearby.",
    "reasoning_and_reflection": "The hand towel is currently on the hanger and not inside the cabinet. Not satisfied.\\n The cabinet is closed, so I must open it before placing the towel inside, and close it again afterward to satisfy the target state.\\n Therefore, I need to retrieve the towel, open the cabinet, store the towel, and then close the cabinet.",
    "language_plan": "Begin by locating the hand towel on the hanger. Once you have found the hand towel, proceed to pick it up. Next, search for the cabinet and identify its location using the \"Find\" action. After finding the cabinet, open it. Place the hand towel inside the cabinet, and finally, close the cabinet door.",
    "executable_plan": [
        {"action_id": 9, "action_name": "find a HandTowel"},
        {"action_id": 89, "action_name": "pick up the HandTowel"},
        {"action_id": 48, "action_name": "find a Cabinet"},
        {"action_id": 145, "action_name": "open the Cabinet"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 146, "action_name": "close the Cabinet"}
    ]
}
''',

'''
Human instruction: Place a microwaved slice of apple on to the white table.
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
    "visual_state_description": "A whole apple lies next to a microwave; a white dining table is visible across the kitchen.",
    "reasoning_and_reflection": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a microwaved slice of apple placed on the white dining table.\\n I can see a whole apple in the scene, but it has not been sliced. Not satisfied.\\n I should heat the slice using the microwave. The microwave is visible, but no slice is currently inside or being heated. Not satisfied.\\n I should then place the heated slice on the white dining table. The table is visible, but it is currently empty. Not satisfied.\\n Therefore, I must slice the apple, microwave a slice, and place the microwaved slice on the white dining table.",
    "language_plan": "Locate an apple and pick it up, find a dining table and put the apple down, look for a knife and pick it up, find another apple and slice it, find the dining table and put the knife down, locate an apple and pick it up, search for a microwave and open it, put the apple in the microwave and close it, turn on the microwave to heat the apple slice, turn off the microwave and open it, pick up the microwaved apple slice, close the microwave, find the white table, and place the microwaved apple slice on the white table.",
    "executable_plan": [
        {"action_id": 22, "action_name": "find a Apple"},
        {"action_id": 107, "action_name": "pick up the Apple"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 34, "action_name": "find a Knife"},
        {"action_id": 127, "action_name": "pick up the Knife"},
        {"action_id": 22, "action_name": "find a Apple"},
        {"action_id": 160, "action_name": "slice the Apple"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 22, "action_name": "find a Apple"},
        {"action_id": 107, "action_name": "pick up the Apple"},
        {"action_id": 38, "action_name": "find a Microwave"},
        {"action_id": 143, "action_name": "open the Microwave"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 144, "action_name": "close the Microwave"},
        {"action_id": 149, "action_name": "turn on the Microwave"},
        {"action_id": 150, "action_name": "turn off the Microwave"},
        {"action_id": 143, "action_name": "open the Microwave"},
        {"action_id": 22, "action_name": "find a Apple"},
        {"action_id": 107, "action_name": "pick up the Apple"},
        {"action_id": 144, "action_name": "close the Microwave"},
        {"action_id": 18, "action_name": "find a DiningTable"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
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
    "visual_state_description": "A loaf of bread sits beside a bread knife on the countertop.",
    "reasoning_and_reflection": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a sliced piece of bread placed on the countertop. I can see a loaf of bread and a knife in the scene, but the bread has not been sliced. Not satisfied.\\n I can also see the countertop, but there is no slice of bread on it. Not satisfied.\\n Therefore, I must use the knife to slice the bread and place one of the slices on the countertop.",
    "language_plan": "First locate a knife and pick it up, find a loaf of bread, and slice the bread. Find a cabinet, open it, and put the knife inside, then close the cabinet. Find a slice of bread, pick it up, find the countertop, and place the slice of bread on the counter.",
    "executable_plan": [
        {"action_id": 34, "action_name": "find a Knife"},
        {"action_id": 127, "action_name": "pick up the Knife"},
        {"action_id": 53, "action_name": "find a Bread"},
        {"action_id": 161, "action_name": "slice the Bread"},
        {"action_id": 48, "action_name": "find a Cabinet"},
        {"action_id": 145, "action_name": "open the Cabinet"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 146, "action_name": "close the Cabinet"},
        {"action_id": 53, "action_name": "find a Bread"},
        {"action_id": 92, "action_name": "pick up the Bread"},
        {"action_id": 78, "action_name": "find a Fridge"},
        {"action_id": 139, "action_name": "open the Fridge"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 140, "action_name": "close the Fridge"},
        {"action_id": 139, "action_name": "open the Fridge"},
        {"action_id": 53, "action_name": "find a Bread"},
        {"action_id": 92, "action_name": "pick up the Bread"},
        {"action_id": 140, "action_name": "close the Fridge"},
        {"action_id": 39, "action_name": "find a CounterTop"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
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
    "visual_state_description": "A dirty pan rests in the sink next to a faucet; the fridge is across the room.",
    "reasoning_and_reflection": "I need to check each part in the target state to see whether it is satisfied.\\n I should have a clean pan placed inside the refrigerator. I can see a pan, but it is currently dirty and sitting in the sink. Not satisfied.\\n The fridge is visible but closed, and the pan is not inside it. Not satisfied.\\n Therefore, I must wash the pan to make it clean, then place it into the refrigerator and close the door.",
    "language_plan": "Locate a pan and pick it up, find a sink, and put the pan down in the sink. Find a faucet, turn it on to clean the pan, and then turn it off. Pick up the cleaned pan, find the refrigerator, open the refrigerator, place the pan inside, and close the refrigerator.",
    "executable_plan": [
        {"action_id": 7, "action_name": "find a Pan"},
        {"action_id": 86, "action_name": "pick up the Pan"},
        {"action_id": 79, "action_name": "find a Sink"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 2, "action_name": "find a Faucet"},
        {"action_id": 155, "action_name": "turn on the Faucet"},
        {"action_id": 156, "action_name": "turn off the Faucet"},
        {"action_id": 7, "action_name": "find a Pan"},
        {"action_id": 86, "action_name": "pick up the Pan"},
        {"action_id": 78, "action_name": "find a Fridge"},
        {"action_id": 139, "action_name": "open the Fridge"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 140, "action_name": "close the Fridge"}
    ]
}
''',

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
    "visual_state_description": "A cellphone lies beside an open cardboard box on the couch; a black desk stands opposite.",
    "reasoning_and_reflection": "I need to check each part in the target state to see whether it is satisfied.\\n First, the phone should be inside the box. Currently, the phone is next to the box on the couch, not inside it. Not satisfied.\\n Second, the box should be on the black desk. Currently, the box is on the couch. Not satisfied.\\n Therefore, I must first place the phone inside the box, then move the box from the couch to the black desk.",
    "language_plan": "First, find the phone, pick it up, locate the box on the couch, place the phone inside the box, pick up the box. Then, use the \"Find\" action to find the black desk, and place the box on the desk.",
    "executable_plan": [
        {"action_id": 12, "action_name": "find a CellPhone"},
        {"action_id": 93, "action_name": "pick up the CellPhone"},
        {"action_id": 61, "action_name": "find a Box"},
        {"action_id": 133, "action_name": "put down the object in hand"},
        {"action_id": 103, "action_name": "pick up the Box"},
        {"action_id": 49, "action_name": "find a Desk"},
        {"action_id": 133, "action_name": "put down the object in hand"}
    ]
}
'''
]
