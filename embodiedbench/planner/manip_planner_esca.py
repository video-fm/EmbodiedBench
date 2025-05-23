import ast
import collections
import torch
import sys
import os.path as osp
import numpy as np
import cv2
import base64
import json
import random
import time
import logging
from mimetypes import guess_type
from embodiedbench.envs.eb_manipulation.eb_man_utils import ROTATION_RESOLUTION, VOXEL_SIZE
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.planner.planner_utils import local_image_to_data_url, template_manip, template_lang_manip
from embodiedbench.evaluator.config.eb_manipulation_example import scene_graph_examples
from embodiedbench.main import logger
import torch
import torch.nn.functional as F
import math
from collections import Counter, defaultdict
import os

model_lib_path = os.getenv('LASER_MODEL_PATH')
if not model_lib_path:
    raise ValueError("Please set the LASER_MODEL_PATH environment variable to the path of your LASER model installation")
sys.path.append(model_lib_path)
from llava_clip_model_v3 import PredicateModel as PredicateModel_v3

VISUAL_ICL_EXAMPLES_PATH = "embodiedbench/evaluator/config/visual_icl_examples/eb_manipulation"
VISUAL_ICL_EXAMPLE_CATEGORY = {
    "pick": "pick_cube_shape",
    "place": "place_into_shape_sorter_color",
    "stack": "stack_cubes_color",
    "wipe": "wipe_table_direction"
}

class ManipPlanner():
    def __init__(self, model_name, model_type, system_prompt, examples, n_shot=0, obs_key='front_rgb', chat_history=False, language_only=False, multiview=False, multistep=False, visual_icl=False, tp=1, kwargs={}):
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to include all the chat history for prompting
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp, task_type='manip')

        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.multi_view = multiview
        self.multi_step_image = multistep
        self.visual_icl = visual_icl


        self.scene_graph_prompt = f'''## Robot Vision Task: Object Identification

You are a robot arm observing a workspace from a bird's-eye view. Objects of interest in the image are labeled with numbers.

**Your Task:**
Identify each object surrounded by a red bounding box or mask by its color and shape. The objects are numbered starting from 1. Return your findings in a **compact JSON format**.

**Instructions:**
1.  **Analyze the Image:** Carefully examine the provided image. Focus on the objects enclosed by red bounding boxes. Do not look at the red bounding box.
2.  **Identify Color and Shape:**
    *   **Color:** Determine the object's actual color, **ignoring the red bounding box**. Choose the color from the allowed list below.
    *   **Shape:** Identify the object's shape. Common shapes include:
        *   `cylinder`: A solid object with **circular top and bottom faces** and straight vertical sides. Looks like a can or pill bottle. **NOT** flat like a rectangle or pointed like a prism.
        *   `container`: A **box-like object that is open at the top**, used to hold other items. Its walls are usually rectangular and may be unequal. **NOT** a cube (which is closed) and **NOT** a flat shape.
        *   `star`: A 3D object with a **top surface that has 5 or more distinct spikes or raised points**. These objects may appear symmetric or compact, but the **presence of spikes** makes them stars.**NOT** a cube — cubes are perfectly square and have **no sharp tips, edges, or raised points**.
        *   `cube`: A **solid block with six equal square faces**. All sides are the same length. **NOT** a container or a star.
        *   `triangular prism`: A solid with **two triangular ends** and **three rectangular sides**. It looks like a 3D triangle. **NOT** a moon (which is curved).
        *   `moon`: A **curved, crescent-like shape** with one flat side. Often looks like a half-circle. **NOT** a triangular prism or any straight-edged shape.
        *   `rectangle`: A flat, thin **4-sided object with 90° corners** and unequal side lengths. Think of strips or mats. **NOT** a cube or container.
        *   `sponge`: A **rounded rectangular object** that appears squishy, like a cleaning sponge or pill. **NOT** a cylinder or cube.
        *   `shape sorter`: An object with **multiple distinct shapes embedded or protruding from its surface** (like triangles, stars, and circles combined). **NOT** a basic shape — label this only if the object clearly contains many mixed shapes.
        *   Other common geometric shapes.
        * If the object has **sharp spikes or starfish-like tips**, it is a `"star"`, not a `"cube"` — even if it looks compact.

3.  **Format the Output:**
    *   For entities, describe each object as: `"<color> <shape>"`
    *   Combine all descriptions into a JSON dictionary under the keys `"shapes", "colors", "entities"`.
    *   **Strict JSON Format:** Ensure your output is a single, compact JSON object with **no unnecessary line breaks or spacing**.

**Allowed Colors:**
["red", "maroon", "lime", "green", "blue", "navy", "yellow", "cyan", "magenta", "silver", "gray", "olive", "purple", "teal", "azure", "violet", "rose", "black", "white", "orange", "brown", "grey"]

**Important Visual Rule:**
- If the object has **any sharp spikes or pointy tips**, it is **NEVER** a `"cube"`.
- Cubes have **smooth, flat, square faces** with **no protrusions or spikes**.
- If you see **raised points or a non-uniform top**, classify as `"star"` — even if it looks symmetric.

**Color Clarification Rules (Important!):**
- **Cyan vs. Teal**:
    - If the color looks **light, bright, or turquoise**, classify it as `"cyan"`.
    - If the color is **dark or dull green-blue**, it is `"teal"`.
    - **NEVER classify a dark greenish-blue object as "cyan"**, even if it has a slight blue tint.
- **White**:
  - A **pure, bright color** with no tint. Looks like **paper or snow**.
  - Use `"white"` **only if the object looks completely colorless and highly bright**.
  - **NOT** slightly metallic or dim.
- **Gray**:
  - A **flat, neutral tone** between black and white.
  - Looks like **ash, concrete, or fog**.
  - **NOT** shiny or reflective.
- **Silver**:
  - A **light gray color with metallic shine or gloss**.
  - Appears **reflective or sparkly** like jewelry or metal parts.
  - If the object has a **metallic look**, it's `"silver"` — **even if it’s nearly white**.

**Guidance Examples:**
(Following this prompt, examples will be provided showing input images and expected JSON outputs to further clarify the task.)
'''

        self.template_manip_sg = ''
        self.scene_graph_examples = scene_graph_examples[:n_shot]

        clip_model_name = "openai/clip-vit-base-patch16"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predicate_model = PredicateModel_v3(hidden_dim = 128, num_top_pairs=1, device=device, model_name=clip_model_name).to(device)

    def load_model(self, model_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_dir, map_location=device, weights_only=False)
        return model

    def process_prompt(self, user_instruction, avg_obj_coord, task_variation, prev_act_feedback=[]):
        user_instruction = user_instruction.rstrip('.')
        scene_graph_prompt = ''

        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '\n'.join([f'Example {i}: \n{x}' for i, x in enumerate(self.examples[task_variation][:self.n_shot])]))
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
                scene_graph_prompt = self.scene_graph_prompt

            task_prompt = f"\n## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "

        elif self.chat_history: ## doesn't go in here
            general_prompt = f'The human instruction is: {user_instruction}.'
            general_prompt += '\n\n The gripper action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                general_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
            task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with the 7-dimsension action.'''

        else:
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '\n'.join([f'Example {i}: \n{x}' for i, x in enumerate(self.examples[task_variation][:self.n_shot])]))
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
                scene_graph_prompt = self.scene_graph_prompt

            task_prompt = f"\n## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
            for i, action_feedback in enumerate(prev_act_feedback):
                task_prompt += f"{action_feedback}, "
                scene_graph_prompt += f"{action_feedback}, "


        obj_coords = ast.literal_eval(avg_obj_coord)


        scene_graph_prompt += f"**Remember FOR EACH of the objects (contained by a red bounding box), predict the shape, color, and entity. There should be {len(obj_coords)} objects so the final length of entities, shapes, and colors should be {len(obj_coords)}.**\n"
        scene_graph_prompt += f"**CRITICAL: For EACH object, verify its combined ENTITY ('<color> <shape>'). These MUST be 100% accurate. PAY SPECIAL ATTENTION to correctly identifying complex shapes like 'star' and 'triangular prism' by recalling their visual definitions provided in the initial instructions. Do NOT misidentify them as simpler shapes (e.g., do NOT mistake a 'star' for a 'cube', or a 'triangular prism' for a 'container'). Utter precision is required for all objects.**\n"
        scene_graph_prompt += f"\nComplete the task for this human instruction: {user_instruction}.\n Input: {avg_obj_coord}\n"

        return general_prompt, task_prompt, scene_graph_prompt



    def process_prompt_visual_icl(self, user_instruction, avg_obj_coord, prev_act_feedback=[]):
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            task_prompt = f"## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
        elif self.chat_history:
            general_prompt = f'The human instruction is: {user_instruction}.'
            general_prompt += '\n\n The gripper action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                general_prompt += '\n Step {}, the output action **{}**, env feedback: {}'.format(i, action_feedback[0], action_feedback[1])
            task_prompt = f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history and environment feedback and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with the 7-dimsension action.'''
        else:
            if self.n_shot >= 1:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            else:
                general_prompt = self.system_prompt.format(VOXEL_SIZE, VOXEL_SIZE, int(360 / ROTATION_RESOLUTION), ROTATION_RESOLUTION, '')
            task_prompt = f"## Now you are supposed to follow the above examples to generate a sequence of discrete gripper actions that completes the below human instruction. \nHuman Instruction: {user_instruction}.\nInput: {avg_obj_coord}\nOutput gripper actions: "
            for i, action_feedback in enumerate(prev_act_feedback):
                task_prompt += f"{action_feedback}, "
        return general_prompt, task_prompt

    def get_message(self, images, prompt, task_prompt, messages=[]):
        if self.language_only and not self.visual_icl:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + task_prompt}],
                }
            ]
        else:
            if self.multi_step_image:
                current_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}],
                    }
                ]

                # use the last three imags as multi-step images
                if len(images) >= 3:
                    multi_step_images = images[-3:-1]
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": "You are given the scene observations from the last two steps:",
                        }
                    )
                    for image in multi_step_images:
                        if type(image) == str:
                            image_path = image
                        else:
                            image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                            cv2.imwrite(image_path, image)
                        data_url = local_image_to_data_url(image_path=image_path)
                        current_message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        )

                    # add the current task prompt and input image
                    current_message[0]["content"].append(
                        {
                            "type": "text",
                            "text": task_prompt,
                        }
                    )

                    # add the current step image
                    current_step_image = images[-1]
                    if type(current_step_image) == str:
                        image_path = current_step_image
                    else:
                        image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                        cv2.imwrite(image_path, current_step_image)
                    data_url = local_image_to_data_url(image_path=image_path)
                    current_message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    )
                else:
                    full_prompt = prompt + task_prompt
                    current_message = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": full_prompt}],
                        }
                    ]

                    for image in images:
                        if type(image) == str:
                            image_path = image
                        else:
                            image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                            cv2.imwrite(image_path, image)

                        data_url = local_image_to_data_url(image_path=image_path)
                        current_message[0]["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url,
                                }
                            }
                        )

            else:
                full_prompt = prompt + task_prompt
                current_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt}],
                    }
                ]

                for image in images:
                    if type(image) == str:
                        image_path = image
                    elif type(image) == list:
                        image_path = image[0]
                    else:
                        image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                        cv2.imwrite(image_path, image)

                    data_url = local_image_to_data_url(image_path=image_path)
                    current_message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }
                    )

            return current_message

    def get_message_visual_icl(self, images, first_prompt, task_prompt, task_variation, messages=[]):
        current_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_prompt}
                ],
            }
        ]
        visual_task_variation = VISUAL_ICL_EXAMPLE_CATEGORY[task_variation.split('_')[0]]
        task_specific_image_example_path = osp.join(VISUAL_ICL_EXAMPLES_PATH, visual_task_variation)
        icl_text_examples = self.examples[task_variation]
        stop_idx = 2
        for example_idx, example in enumerate(icl_text_examples):
            if example_idx >= stop_idx:
                break
            current_image_example_path = osp.join(task_specific_image_example_path, f"episode_{example_idx+1}_step_0_front_rgb_annotated.png")
            example = "Example {}:\n{}".format(example_idx+1, example)
            data_url = local_image_to_data_url(image_path=current_image_example_path)

            # Add the example image and the corresponding text to the message
            current_message[0]["content"].append(
                {
                    "type": "text",
                    "text": example,
                }
            )
            current_message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    }
                }
            )
        # add the task prompt
        current_message[0]["content"].append(
            {
                "type": "text",
                "text": task_prompt,
            }
        )

        for image in images:
            if type(image) == str:
                image_path = image
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            data_url = local_image_to_data_url(image_path=image_path)
            current_message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    }
                }
            )
        return current_message

    def json_to_action(self, output_text):
        try:
            json_object = json.loads(output_text)
            action = []
            try:
                executable_plan = json_object['executable_plan'] if 'executable_plan' in json_object else json_object["properties"]["executable_plan"]
            except:
                print("Failed to get executable plan from json object")
                print('random action')
                self.output_json_error += 1
                arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)]
                gripper = [1.0]  # Always open
                action = arm + gripper
                return [action], None
            if type(executable_plan) == str:
                try:
                    executable_plan = ast.literal_eval(executable_plan)
                except Exception as e:
                    print("Failed to decode string executable plan to list executable plan:", e)
                    print('random action')
                    self.output_json_error += 1
                    arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)]
                    gripper = [1.0]  # Always open
                    action = arm + gripper
                    return [action], None
            if len(executable_plan) > 0:
                for x in executable_plan:
                    if type(x) == tuple:
                        x = list(x)
                    if 'action' in x:
                        list_action = x['action']
                    else:
                        if type(x) == list and type(x[0]) == int:
                            list_action = x
                        elif 'action' in x[0]:
                            list_action = x[0]["action"]
                        else:
                            list_action = x
                    if type(list_action) == str:
                        try:
                            list_action = ast.literal_eval(x['action'])
                        except Exception as e:
                            print("Failed to decode string action to list action:", e)
                            print('random action')
                            action = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)] + [1.0]
                            self.output_json_error += 1
                            return [action], None
                    action.append(list_action)
                return action, json_object
            else:
                print("Empty executable plan, quit the episode ...")
                self.output_json_error = -1
                return [], output_text
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            print('random action')
            self.output_json_error += 1
            arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)]
            gripper = [1.0]  # Always open
            action = arm + gripper
            return [action], None
        except Exception as e:
            print("An expected error occurred:", e)
            print('random action')
            self.output_json_error += 1
            arm = [random.randint(0, VOXEL_SIZE) for _ in range(3)] + [random.randint(0, (360 / ROTATION_RESOLUTION) - 1) for _ in range(3)]
            gripper = [1.0]  # Always open
            action = arm + gripper
            return [action], None

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
        out = self.model.respond(prompt, obs)
        out = out.replace("'",'"')
        out = out.replace('\"s ', "\'s ")
        out = out.replace('```json', '').replace('```', '')
        logger.debug(f"Model Output:\n{out}\n")
        action, _ = self.json_to_action(out)
        self.planner_steps += 1
        return action, out

    def process_laser_inputs(self, obs, bboxes, masks, cate_kws, unary_kws):
        image_path = obs[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        new_bboxes = []
        for bbox in bboxes:
            new_bboxes.append({'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]})

        if len(masks) > 0:
            object_ids = [[0,0,i] for i in range(len(masks))]
        else:
            object_ids = [[0,0,i] for i in range(len(new_bboxes))]

        ## reshape each mask
        new_masks = []
        for mask in masks:
            new_masks.append(mask[:, :, None])

        ## cast bounding boxes to integers
        new_bboxes = [[int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])] for bbox in new_bboxes]

        return {
            'batched_video_ids': [0],
            'batched_videos': [image],
            'batched_masks': new_masks,
            'batched_bboxes': new_bboxes,
            'batched_names': [cate_kws],
            'batched_object_ids': object_ids,
            'batched_unary_kws': [unary_kws],
            'batched_binary_kws': [[]],
            'batched_obj_pairs': [],
            'batched_video_splits': [0],
            'batched_binary_predicates': [None],
        }

    def process_laser_outputs(self, cate_probs, unary_probs, binary_probs, color_to_shape_dict, shape_to_idx):
        idx_to_name_prob = {}
        idx_to_color_prob = {}
        output_string = ""

        ## COLORS
        for (_, item_idx, item_string), prob in unary_probs[0].items():
            if item_idx in idx_to_color_prob:
                old_string, old_prob = idx_to_color_prob[item_idx]
                if prob > old_prob:
                    idx_to_color_prob[item_idx] = (item_string, prob)
            else:
                idx_to_color_prob[item_idx] = (item_string, prob)

        for item_idx, (color, prob) in idx_to_color_prob.items():
            if len(color_to_shape_dict[color]) > 1:
                shapes = list(color_to_shape_dict[color])

                ## compare the probabilities of the shapes
                max_prob = 0.0
                max_prob_obj = ""

                for shape in shapes:
                    cur_prob = cate_probs[0][(item_idx, shape)].item()
                    if cur_prob > max_prob:
                        max_prob = cur_prob
                        max_prob_obj = shape
                output_string += f"Object {item_idx + 1} is {color} {max_prob_obj}. "

            else:
                output_string += f"Object {item_idx + 1} is {color} {list(color_to_shape_dict[color])[0]}. "



        return output_string


    def compute_entropy_from_logits(self, valid_logits):
        """
        Given a list of (name, logit), compute softmax and entropy.
        Returns (entropy, softmax_probs_dict)
        """
        if not valid_logits:
            return float('inf'), {}

        names, logits = zip(*valid_logits)
        logits_tensor = torch.tensor(logits)
        probs = torch.softmax(logits_tensor, dim=0).tolist()

        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        softmax_probs = dict(zip(names, probs))
        return entropy, softmax_probs

    def assign_entities_with_dynamic_softmax(self, entity_list, predicted_logits):
        """
        Assigns entity names to object ids using dynamic entropy and softmax on logits.

        Args:
            entity_list (List[str]): List of target entity names (with possible repetitions).
            predicted_logits (List[((int, str), torch.Tensor)]): List of ((obj_id, name), logit_tensor) pairs.

        Returns:
            Dict[int, str]: Mapping from obj_id to assigned entity name.
        """
        name_budget = Counter(entity_list)
        obj_ids = sorted(set(obj_id for (obj_id, _), _ in predicted_logits.items()))

        # Group raw logits by object
        logits_per_obj = defaultdict(list)
        for (obj_id, name), logit in predicted_logits.items():
            logits_per_obj[obj_id].append((name, logit.item()))

        assigned = {}

        while len(assigned) < len(obj_ids):
            entropy_list = []
            prob_per_obj = {}

            # Compute entropy for each unassigned obj using current valid choices
            for obj_id in obj_ids:
                if obj_id in assigned:
                    continue
                valid_logits = [
                    (name, logit)
                    for name, logit in logits_per_obj[obj_id]
                    if name_budget[name] > 0
                ]
                entropy, softmax_probs = self.compute_entropy_from_logits(valid_logits)
                entropy_list.append((entropy, obj_id))
                prob_per_obj[obj_id] = softmax_probs

            # Choose object with lowest entropy
            _, chosen_obj = min(entropy_list)
            softmax_probs = prob_per_obj[chosen_obj]

            # Pick the highest-softmax-prob name that's still valid
            available_names = [(name, prob) for name, prob in softmax_probs.items() if name_budget[name] > 0]
            if not available_names:
                # raise ValueError(f"No valid candidates left for object {chosen_obj}")
                break

            best_name = max(available_names, key=lambda x: x[1])[0]

            assigned[chosen_obj] = best_name
            name_budget[best_name] -= 1

        return assigned


    def act(self, observation, user_instruction, avg_obj_coord, task_variation, bboxes, masks):
        # observation = ['running/eb_manipulation/InternVL2_5-38B-MPO/baseline/base/images/episode_1/episode_1_step_0_front_rgb.png']
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path

        if self.visual_icl and not self.language_only: ## usually doesn't go in here so don't touch this
            first_prompt, task_prompt = self.process_prompt_visual_icl(user_instruction, avg_obj_coord, prev_act_feedback=self.episode_act_feedback)
            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL-72B-Instruct' in self.model_name:
                task_prompt += "\n\n"
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation, self.episode_messages)
                else:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)
        else:
            full_example_prompt, task_prompt, scene_graph_prompt = self.process_prompt(user_instruction, avg_obj_coord, task_variation, prev_act_feedback=self.episode_act_feedback)
            if 'gemini' in self.model_name or 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL-72B-Instruct' in self.model_name:
                task_prompt += "\n\n"
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt, self.episode_messages)
                else:
                    self.episode_messages = self.get_message(obs, full_example_prompt, task_prompt)


        scene_graph_prompt += "\n\n"
        scene_graph_prompt = scene_graph_prompt + template_lang_manip if self.language_only else scene_graph_prompt + self.template_manip_sg
        scene_graph_prompt += "\n\n"
        scene_graph_prompt += "**Required JSON Output Structure Example:**\n"
        scene_graph_prompt += "- The output json should have exactly THREE KEYS: 'entities', 'shapes', 'colors'.\n"
        scene_graph_prompt += "{{'entities': ['blue cube', 'yellow cylinder', 'green triangular prism'], 'shapes': ['cube', 'cylinder', 'triangular prism'], 'colors': ['blue', 'yellow', 'green']}}\n"

        ##### SCENE GRAPH GENERATION #####
        # scene_graph_msg = self.get_message(obs, scene_graph_prompt, "", messages=self.episode_messages)
        scene_graph_msg = self.get_message(obs, scene_graph_prompt, "")
        try:
            out = self.model.respond(scene_graph_msg, get_scene_graph=True)
        except Exception as e:
            print(e)

        output = json.loads(out)

        ## construct inputs to laser
        shape_kws = output['shapes']
        color_kws = output['colors']
        entities_kws = output['entities']
        laser_input_dict = self.process_laser_inputs(obs, bboxes, masks, entities_kws, color_kws)

        if len(bboxes) != 0 and len(masks) != 0:

            entities_logits, color_logits, binary_logits, _ = self.predicate_model(**laser_input_dict, output_logit=True)

            result = self.assign_entities_with_dynamic_softmax(entities_kws, entities_logits[0])

            output_string = ", 'Current visual state description': 'From left to right, I can see "
            result = sorted(result.items(), key=lambda x: x[0])
            obj_coords = ast.literal_eval(avg_obj_coord)
            for i, (object_idx, object_name) in enumerate(result):
                output_string += f"a {object_name} at {obj_coords[f'object {object_idx + 1}']}, "
            output_string = output_string[:-2] + ".'}"

            print("output_string: ", output_string)
            part1 = self.episode_messages[0]['content'][0]['text'].split("Output gripper actions")[0]
            part2 = self.episode_messages[0]['content'][0]['text'].split("Output gripper actions")[1]
            new_prompt = part1[:-2] + output_string + part2

            #### Change the prompt for the model
            self.episode_messages[0]['content'][0]['text'] = new_prompt

        if self.model_type == 'custom': ## doesn't go in here
            return self.act_custom(full_example_prompt + task_prompt + "\n\n" + template_manip, obs[0])

        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try:
                out = self.model.respond(self.episode_messages)
            except:
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            try:
                out = self.model.respond(self.episode_messages)
            except:
                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )

        logger.debug(f"Model Output:\n{out}\n")
        self.planner_steps += 1
        action, json_output = self.json_to_action(out)
        return action, out

    def update_info(self, info):
        env_feedback = info['env_feedback']
        action = info['action']
        self.episode_act_feedback.append([action, env_feedback])

