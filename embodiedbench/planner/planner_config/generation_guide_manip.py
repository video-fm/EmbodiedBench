vlm_generation_guide_manip={
    "type": "object",
    'properties': {
        "visual_state_description": {
            "type": "string",
            "description": "Describe the color and shape of each object in the detection box in the numerical order in the image. Then provide the 3D coordinates of the objects chosen from input.",
        },
        "reasoning_and_reflection": {
            "type": "string",
            "description": "Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available.",
        },
        "language_plan": {
            "type": "string",
            "description": "A list of natural language actions to achieve the user instruction. Each language action is started by the step number and the language action name.",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of discrete actions needed to achieve the user instruction, with each discrete action being a 7-dimensional discrete action.",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The 7-dimensional discrete action in the format of a list given by the prompt",
                    }
                },
                "required": ["action"]
            }
        },
    },
    "required": ["visual_state_description", "reasoning_and_reflection", "language_plan", "executable_plan"]
}

llm_generation_guide_manip={
    "type": "object",
    'properties': {
        "reasoning_and_reflection": {
            "type": "string",
            "description": "Reason about the overall plan that needs to be taken on the target objects, and reflect on the previous actions taken if available.",
        },
        "language_plan": {
            "type": "string",
            "description": "The list of actions to achieve the user instruction. Each action is started by the step number and the action name.",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of actions needed to achieve the user instruction, with each action being a 7-dimensional discrete action in the format of a list.",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The 7-dimensional discrete action in the format of a list given by the prompt",
                    }
                },
                "required": ["action"]
            }
        },
    },
    "required": ["reasoning_and_reflection", "language_plan", "executable_plan"]
}

manip_vlm_sg_generation_guide={
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "description": "A list of entities in the scene graph.",
            "items": {
                "type": "string",
                "description": "The name of the entity."
            }
        },
        "shapes": {
            "type": "array",
            "description": "A list of shapes of the entities.",
            "items": {
                "type": "string",
                "description": "The shape of the object in the numerical order in the image."
            }
        },
        "colors": {
            "type": "array",
            "description": "A list of colors of the entities.",
            "items": {
                "type": "string",
                "description": "The color of the object in the numerical order in the image."
            }
        }
    },
}
