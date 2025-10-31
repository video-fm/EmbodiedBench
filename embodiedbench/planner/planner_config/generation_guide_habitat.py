vlm_generation_guide={
    "type": "object",
    'properties': {
        "visual_state_description": {
            "type": "string",
            "description": "Description of current state from the visual image",
        },
        "reasoning_and_reflection": {
            "type": "string",
            "description": "summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task",
        },
        "language_plan": {
            "type": "string",
            "description": "The list of actions to achieve the user instruction. Each action is started by the step number and the action name",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of actions needed to achieve the user instruction, with each action having an action ID and a name. Do not output empty list.",
            "items": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "integer",
                        "description": "The action ID to select from the available actions given by the prompt",
                    },
                    "action_name": {
                        "type": "string",
                        "description": "The name of the action",
                    }
                },
                "required": ["action_id", "action_name"],
            }
        },
    },
    "required": ["visual_state_description", "reasoning_and_reflection", "language_plan", "executable_plan"]
}

llm_generation_guide={
    "type": "object",
    'properties': {
        "reasoning_and_reflection": {
            "type": "string",
            "description": "summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task",
        },
        "language_plan": {
            "type": "string",
            "description": "The list of actions to achieve the user instruction. Each action is started by the step number and the action name",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of actions needed to achieve the user instruction, with each action having an action ID and a name. List length should not exceed 20.",
            "items": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "integer",
                        "description": "The action ID to select from the available actions given by the prompt",
                    },
                    "action_name": {
                        "type": "string",
                        "description": "The name of the action",
                    }
                },
                "required": ["action_id", "action_name"]
            }
        },
    },
    "required": ["reasoning_and_reflection", "language_plan", "executable_plan"]
}

vlm_sg_generation_guide = {
    "type": "object",
    "properties": {
        "visual_state_description": {
            "type": "string",
            "description": "Description of the scene from the visual input",
        },
        "scene_graph_info": {
            "type": "object",
            "properties": {
                "target_state": {
                    "type": "string",
                    "description": "Natural language description of the desired final state of the scene",
                },
                "target_entities": {
                    "type": "array",
                    "description": "List of object and location identifiers involved in the target state",
                    "items": {
                        "type": "string"
                    }
                },
                "target_relations": {
                    "type": "array",
                    "description": "List of [subject, relation, object] defining the desired relationships between entities",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 3,
                        "maxItems": 3
                    }
                },
                "current_state": {
                    "type": "string",
                    "description": "Natural language description of the current state of the scene",
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of why the current state is not the target state and what changes are needed",
                },
                "current_entities": {
                    "type": "array",
                    "description": "List of object and location identifiers currently present in the scene",
                    "items": {
                        "type": "string"
                    }
                },
                "current_relations": {
                    "type": "array",
                    "description": "List of [subject, relation, object] describing the current relationships between entities",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    }
                }
            },
            "required": [
                "target_state",
                "target_entities",
                "target_relations",
                "current_state",
                "explanation",
                "current_entities",
                "current_relations"
            ]
        },
    },
    "required": ["visual_state_description", "scene_graph_info"],
}


vlm_target_generation_guide = {
        "type": "object",
        "parameters": {
            "target_name": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "The main object(s) of interest"
                },
                "description": "List of names representing the target entity"
            },
            "blocking_name": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Names of objects that block or obscure the target"
                },
                "description": "List of object names that are considered blocking"
            },
            "target_attribute": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Attributes or spatial details about the target"
                },
                "description": "Attributes associated with the target entity"
            },
            "target_relation": {
                "type": "array",
                # "items": {
                #     "type": "array",
                #     "items": {
                #         "type": "string"
                #     },
                #     "minItems": 3,
                #     "maxItems": 3,
                #     "description": "A relation in the form [subject, relation, object]"
                # },
                "items": {
                    "type": "string",
                    "description": "A relation in the form [subject, relation, object]"
                },
                "description": "Triplets describing relationships involving the target"
            },
            "related_objects": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Objects mentioned in relation to the target"
                },
                "description": "Other objects that co-occur with or are related to the target"
            }
        },
    "additionalProperties": False,
    "required": ["target_name", "blocking_name", "target_attribute", "target_relation", "related_objects"]
}
