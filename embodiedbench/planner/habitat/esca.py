import torch
import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_utils import local_image_to_data_url, template, template_lang, fix_json
from embodiedbench.planner.habitat.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger
from PIL import Image
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

MESSAGE_WINDOW_LEN = 5
scene_graph_prompt_path = os.path.join(os.path.dirname(__file__), '../../evaluator/habitat/prompts/habitat_scene_graph_prompts_laser.txt')

import cv2
import matplotlib.pyplot as plt

from groundingdino.util.inference import Model as gd_Model
from groundingdino.util.utils import get_phrases_from_posmap
import torch
import math

from laser.models.llava_clip_model_v3 import PredicateModel

def load_laser_model(model_dir, model_name, epoch, device):
    model_name = model_name + f'.{epoch}.model'
    predicate_model = torch.load(os.path.join(model_dir, model_name), map_location=device)
    return predicate_model

def clean_name(input_name):
    if '_' in input_name:
        input_name_ls = input_name.split('_')
        if input_name_ls[-1].isdigit():
            new_name = ' '.join(input_name_ls[:-1])
        else:
            new_name = ' '.join(input_name_ls)
    else:
        new_name = input_name
    return new_name

def format_scene_graph(target_objects, target_attributes, target_relations):
     # Format structured target state
    if len(target_objects) > 0:
        clean_obj = [target_obj for target_obj in target_objects ]
        object_str = f"The objects are: {clean_obj}"
    else:
        object_str = ""
    
    if len(target_attributes) > 0:
        attr_str_ls = []
        for attr_info in target_attributes:
            if len(attr_info) != 2:
                continue
            attr, obj = attr_info
            attr_str_ls.append(f"({attr}, {obj})")
            
        attr_str = (
            "The attributes are: [" +
            ", ".join(attr_str_ls)
            + "]" if target_attributes else "The attributes are: []"
        )
    else:
        attr_str = ""
        
    rel_str_ls = []
    rel_str = ""
    if len(target_relations) > 0:
        
        for target_info in target_relations:
            if len(target_info) != 3: 
                continue
        
            subj, pred, obj = target_info
            rel_str_ls.append(f"({subj}, {pred}, {obj})")
            
        rel_str = (
            "The relations are: [" +
            ", ".join(rel_str_ls)
            + "]" if target_relations else "The relations are: []"
        )
    else:
        rel_str = ""
    
    output_str_ls = [object_str, attr_str,  rel_str]
    output_str_ls = [i for i in output_str_ls if not i == ""]
    output_str = " ".join(output_str_ls)
    return output_str


def format_state_descriptions(sg_info, current_rel):
    # Extract basic elements
    target_desc = sg_info.get("target_state", "No target state description provided.")
    current_desc = sg_info.get("current_state", "No current state description provided.")
    
    current_rel = [rel for _, _, rel in current_rel]

    sg_str = ""

    sg_str += (f"\n\n The target state: {target_desc}\n" f"The current state: {current_desc}\n")
    
    return sg_str
    

def get_topk_per_object_name(aggr_res, avg_aggr_res, k):
    name_to_ids = defaultdict(list)

    # Group object ids by object name
    for obj_id, content in aggr_res.items():
        if "name" not in content:
            continue
        prob, name = content["name"]
        if name == "other":
            continue
        name_to_ids[name].append(obj_id)

    # Collect top-k per name
    topk_oid_to_score = {}
    for name, obj_ids in name_to_ids.items():
        valid_ids = [(obj_id, avg_aggr_res[obj_id]) for obj_id in obj_ids if obj_id in avg_aggr_res]
        sorted_ids = sorted(valid_ids, key=lambda x: x[1], reverse=True)[:k]
        for obj_id, score in sorted_ids:
            topk_oid_to_score[obj_id] = score

    return topk_oid_to_score

import heapq
from itertools import combinations

def get_topk_per_relation(predicted_name, binary_predicates, top_k):
    # Map from relation -> list of (score, (id1, relation, id2), (name1, name2))
    relation_to_scored_pairs = defaultdict(list)

    for (relation, obj1_label, obj2_label) in binary_predicates:
        for id1, id2 in combinations(predicted_name.keys(), 2):
            if id1 == id2:
                continue
            
            preds1 = predicted_name.get(id1, [])
            preds2 = predicted_name.get(id2, [])

            score1 = next((p for p, label in preds1 if label == obj1_label), 0.0)
            score2 = next((p for p, label in preds2 if label == obj2_label), 0.0)

            combined_score = score1 * score2
            if combined_score > 0:
                relation_to_scored_pairs[relation].append((combined_score, (id1, relation, id2), (obj1_label, obj2_label)))

    # For each relation, keep top-k entries
    topk_per_relation = {}
    for rel, scored_list in relation_to_scored_pairs.items():
        topk = heapq.nlargest(top_k, scored_list, key=lambda x: x[0])
        topk_per_relation[rel] = [
            {"pair": pair, "score": score, "names": (subj, obj)}
            for score, pair, (subj, obj) in topk
        ]

    return topk_per_relation

def merge_object_pair_ids(topk_per_relation): 
    all_object_pairs = set()
    
    for rel, topk_res_ls in topk_per_relation.items():
        for topk_res in topk_res_ls:
            subj, _, obj = topk_res['pair']
            all_object_pairs.add((0, 0, (subj, obj)))

    return sorted(list(all_object_pairs))

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

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    return result if result.endswith(".") else result + "."

def get_valid_bbox_val(v, b):
    return min(max(0, math.floor(v)), b)

def post_process_detections(det, classes):
    detections = det[0]
    class_ids = [0 if class_id is None else class_id for class_id in detections.class_id]
    det.class_names = [classes[class_id] for class_id in class_ids]
    detections.xyxy = np.concatenate((detections.xyxy, det.xyxy), axis=0)
    detections.class_names = detections.class_names + det.class_names
    return detections

def list_depth(lst):
    """Calculates the depth of a nested list."""
    if not (isinstance(lst, list) or isinstance(lst, torch.Tensor)):
        return 0
    elif (isinstance(lst, torch.Tensor) and lst.shape == torch.Size([])) or (isinstance(lst, list) and len(lst) == 0):
        return 1
    else:
        return 1 + max(list_depth(item) for item in lst)

def normalize_prompt(points, labels):
    if list_depth(points) == 3: 
        points = torch.stack([p.unsqueeze(0) for p in points])
        labels = torch.stack([l.unsqueeze(0) for l in labels])
    return points, labels

# modify this function to take in a bbox label and show it
def show_box(box, ax, object_id, label='hi'):
    if len(box) == 0:
        return
    cmap = plt.get_cmap("gist_rainbow")
    cmap_idx = 0 if object_id is None else object_id
    color = list(cmap((cmap_idx * 47) % 256))
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=.5))
    if label is not None:
        ax.text(x0, y0 - 5, label, color='white', fontsize=10, alpha=color[3])
    
def save_prompts_one_image(frame_image, boxes, points, labels, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')
    points, labels = normalize_prompt(points, labels)
    if type(boxes) == torch.Tensor:
        for object_id, (box, label) in enumerate(zip(boxes, labels)):
            if box is not None:
                show_box(box.cpu(), ax, object_id=object_id, label=label)
    elif type(boxes) == dict:
        for object_id, box in boxes.items():
            if box is not None:
                show_box(box.cpu(), ax, object_id=object_id)
    elif type(boxes) == list and len(boxes) == 0:
        pass
    else:
        raise Exception()
    # Show the plot
    plt.savefig(save_path)
    plt.close()

def save_mask_contours_and_boxes_one_image(frame_image, masks, boxes, save_path, labels, sg_type):
    """
    Draws contours and bounding boxes on a copy of the input image and saves it to disk.

    Args:
        frame_image (np.ndarray): The RGB image (H x W x 3).
        masks (list[np.ndarray] or dict[int, np.ndarray]): List or dict of binary masks.
        boxes (list or np.ndarray): List of boxes, each box as [x0, y0, x1, y1].
        save_path (str): Where to save the final image.
    """

    if isinstance(masks, list):
        masks = {i: mask for i, mask in enumerate(masks)}
    contour_img = frame_image.copy()

    # Draw contours
    for obj_id, mask in masks.items():
        segmentation = mask.astype(np.uint8)
        contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, color=(0, 255, 0), thickness=2)

    # Draw bounding boxes
    for object_id, (box, label) in enumerate(zip(boxes, labels)):
        x0, y0, x1, y1 = map(int, box)
        text = label
        org = (x0, y0-5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255) # White color (BGR)
        thickness = 1
        line_type = cv2.LINE_AA

        cv2.putText(contour_img, text, org, font, font_scale, color, thickness, line_type)

    # Save the result
    cv2.imwrite(save_path, cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))

###################

class VLMPlanner():
    def __init__(self, model_name, model_type, actions, system_prompt, examples, n_shot=0, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1, 
                scene_graph_examples = '',
                gd_config_path='', gd_checkpoint_path='', kwargs={}, sg_type="GD",
                laser_model_dir='', laser_model_name='',
                ):

        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.scene_graph_examples = [json.dumps(i) for i in scene_graph_examples[:n_shot]]
        self.sg_type = sg_type
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.set_actions(actions)
        self.model_type = model_type

        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')

        self.scene_graph_prompt = f"## You are a robot operating in a home. Given a task and an image, you should observe the environment to accomplish the task using a defined set of actions." + \
            f" The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} of the environment you will operate in. \n" + \
            open(scene_graph_prompt_path, 'r').read()

        device = "cuda" if torch.cuda.is_available() else "cpu" 

        #####################
        ## GroundingDINO
        self.grounding_model = gd_Model(
            model_config_path=gd_config_path, 
            model_checkpoint_path=gd_checkpoint_path
        )

        self.grounding_model.model = self.grounding_model.model.to(device)

        #####################
        ## LASER set up
        laser_epoch = 0
        model_info = load_laser_model(laser_model_dir, laser_model_name, laser_epoch, device)
        self.predicate_model = PredicateModel(hidden_dim = 0, num_top_pairs=10, device=device, model_name="openai/clip-vit-base-patch32").to(device)
        self.predicate_model.load_state_dict(model_info)

        #####################
        ## ESCA parameters
        self.sg_response = ''
        self.aggr_thres = 0.1
        self.k = 1
        self.rel_thres = 0.3

        print("GD ckpt path", gd_checkpoint_path)
        print("GD config path", gd_config_path)
        print("LASER model dir", laser_model_dir)
        print("LASER model name", laser_model_name)

        self.common_objects = ["wall", "door", "cabinet", "stove", "counter", "window", "other"]


    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str


    def process_prompt(self, user_instruction, prev_act_feedback=[], scene_graph=None):
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])

            if f'---\n## Now the human instruction is: {user_instruction}.' not in prompt:
                prompt += f'\n\n---\n## Now the human instruction is: {user_instruction}.'
            if f'---\n## Now the human instruction is: {user_instruction}.' not in scene_graph_prompt:
                scene_graph_prompt += f'\n\n---\n## Now the human instruction is: {user_instruction}.'

            if self.language_only:
                prompt += f" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            else:
                prompt += f" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."

        elif self.chat_history:
            prompt = f'The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        else:
            if self.n_shot >= 1:
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            if f'---\n## Now the human instruction is: {user_instruction}.' not in prompt:
                prompt += f'\n\n---\n## Now the human instruction is: {user_instruction}.'
            if f'---\n## Now the human instruction is: {user_instruction}.' not in scene_graph_prompt:
                scene_graph_prompt += f'\n\n---\n## Now the human instruction is: {user_instruction}.'

            prompt += "\n\n---\n"

            prompt += "Now, below is the history of interactions you already had with the environment so far. Observe the action history and pay close attention to 1) WHY your actions failed and 2) WHERE you have already searched for certain objects. "
            prompt += "Based on these understanding, generate a new plan to accomplish the task. You MUST NEVER repeat the same invalid actions, and you should not need to search the same place more than once for the same object. Its fine if you are visiting the same place twice for different objects."
            prompt += "You should understand that you have limited number of tries. So, if you are having trouble finding an object, it is more worthwhile to explore more than to try the same actions at the same location more than once."
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])

                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

        return prompt, scene_graph_prompt
    

    @staticmethod
    def convert_to_natural_language(scene_graph):
        out = ""

        out += "---\nNext, below is a natural language description of identified objects. Use these information in conjunction with what you see in the image."
        
        current_objects = scene_graph['current_entities']
        current_relations = {i[0]: i[1] + " " + i[2] for i in scene_graph['current_relations'] if len(i) == 3}

        target_objects = scene_graph['target_entities']
        target_relations = {i[0]: i[1] + " " + i[2] for i in scene_graph['target_relations'] if len(i) == 3}

        target_objects_not_visible = [obj for obj in target_objects if not current_objects]

        ## Current Visible Objects
        for e, entity in enumerate(current_objects):
            out += f"Your currently visible object no. {e+1} is {entity}"

            if entity in current_relations:
                out += f". This object is currently visible {current_relations[entity]}"
            out += ".\n"

        ## Target Objects Not Visible
        for entity in target_objects_not_visible:
            out += f"Your target object {entity} is currently not visible. \n"

        ## Target Objects
        for e, entity in enumerate(target_objects):
            out += f"One of your target objects is {entity}"

            if entity in target_relations:
                out += f". This object needs to be {target_relations[entity]} in order to achieve the task."
            out += ".\n"

        out += "If you are unable to pick up the object at the said location, you should move to a different location immediately. Sometimes objects look similar that may confuse you to think that the target object is in a given location. If the environment rejects your action, you should continue your search somewhere else immediately.\n"
        out += "Different from similar looking objects, there are also similar types of target objects. Note the subtle differences between similar objects like mug and cup. These subtle differences make a difference so ensure that you are manipulating the exact object as required in the task and not something that is similar\n"
        out += "If the identified location of the object is not navigatable, you should infer the nearest navigatable location to pick up the object.\n"
        out += "Lastly, your action should not end with holding an object. If the task is to get something off some place, you need to pick up the object and place it at some other location in order for the task to be successfully completed\n"
        out += "REMEMBER: there is no such thing as repositioning yourself in the same navigatable location to make yourself near enough to the object to pick it up. If you cannot pick up the object at the location, that simply means that location does not have the object and you MUST navigate to another location to pick up the object!\n"
        out += "ALWAYS REMEMBER!!! THERE IS NO SUCH THING AS COMPLETING EARLY. YOU SHOULD NEVER CONCLUDE THAT THE TASK HAS BEEN COMPLETED BECAUSE THE ENVIRONMENT WILL TELL YOU. YOU SHOULD NEVER EVER CONCLUDE THAT THE TASK IS COMPLETED UNLESS THE ENVIROBNMENT SAYS SO."

        return out
    
    def get_sg_kws(self, sg_info):

        target_objects = sg_info['target_entities']
        target_relations = sg_info['target_relations']
        current_objects = sg_info['current_entities']
        current_relations = sg_info['current_relations']
        
        target_names = list(set([clean_name(object) for object in target_objects]))
        current_names = list(set([clean_name(object) for object in current_objects]))

        target_relation_preds = set()
        current_relation_preds = set()
        target_relation_kws = set()
        
        for rel in target_relations:
            if not len(rel) == 3:
                continue
            (s, r, o) = rel
            s = clean_name(s)
            o = clean_name(o)
            target_relation_preds.add((r, s, o))
            target_relation_kws.add(r)
            
        for rel in current_relations:
            if not len(rel) == 3:
                continue
            (s, r, o) = rel
            s = clean_name(s)
            o = clean_name(o)
            current_relation_preds.add((r, s, o))

        query_objects = list(set(target_names + current_names + self.common_objects)) 
        query_relation_preds = list(target_relation_preds.union(current_relation_preds))
        query_other_preds = list(set([("other", sub, obj) for (_, sub, obj) in query_relation_preds]))
        query_relation_preds = query_relation_preds + query_other_preds
        query_relation_kws = list(set([r for (r, _, _) in query_relation_preds]))
        
        return target_names, current_names, query_objects, \
               target_relation_preds, current_relation_preds, \
               target_relation_kws, query_relation_kws, query_relation_preds


    def get_message(self, image, prompt, messages=[], scene_graph=None):
        if scene_graph is not None:
            prompts = prompt.split("---\n## Now the human instruction is:")
            prompts[0] = prompts[0] + self.convert_to_natural_language(scene_graph) + "\n\n"
            prompt = "---\n## Now the human instruction is:".join(prompts)
        if self.language_only:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}],
                }
            ]
        else:
            if type(image) == str:
                image_path = image 
            elif type(image) == list:
                image_path = image[-1]
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                np_image = image
                
                if isinstance(np_image, torch.Tensor):
                    # Move to CPU (if on GPU), detach from graph, and convert to numpy
                    # Also check shape, e.g. if [C,H,W], permute to [H,W,C]
                    if np_image.ndim == 3 and np_image.shape[0] <= 4:  # typical [C,H,W]
                        np_image = np_image.permute(1, 2, 0)
                    np_image = np_image.detach().cpu().numpy()
                    
                    # Make sure it's a copy and thus writable
                    np_image = np_image.copy()

                if isinstance(np_image, Image.Image):
                    # Convert from PIL to numpy array
                    np_image = np.array(image)
                    
                    # Optional: If `image` is in RGB and you want OpenCV's default BGR:
                    # np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                    
                cv2.imwrite(image_path, np_image)

            if self.multistep: # handle multiple images
                i = image_path.split('step_')[-1].strip('.png')
                ind = int(i) if i.isdigit() else int(i.strip('_bbox'))
                content = [{"type": "text", "text": prompt}]
                for i in range(max(ind - self.multistep + 1, 0), ind +1):
                    temp_path = ''.join(image_path.split('step_')[:-1])+ f'step_{str(i)}.png'
                    temp_data_url = local_image_to_data_url(image_path=temp_path)
                    content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": temp_data_url,
                            }})
            else:
                data_url = local_image_to_data_url(image_path=image_path)
                content = [{ "type": "image_url", "image_url": { "url": data_url,}}, {"type": "text", "text": prompt}]

            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action
    
    def json_to_action(self, output_text, json_key='executable_plan'):
        try:
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            if not len(action):
                print('empty plan, stop here')
                action = -2
            else:
                # keep action valid
                for i, act in enumerate(action):
                    if act >= len(self.actions) or act < 0:
                        print('found invlid action')
                        if i == 0:
                            action = -1
                        else:
                            action = action[:i]
                        break
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            self.output_json_error += 1
            action = -1
        return action

    def predict_gd(self, image, classes, box_threshold=0.25, text_threshold=0.3):
        caption = ". ".join(classes)
        processed_caption = preprocess_caption(caption)
        processed_image = self.grounding_model.preprocess_image(image_bgr=image).to(self.grounding_model.device)
        with torch.no_grad():
            outputs = self.grounding_model.model(processed_image.unsqueeze(0), captions=processed_caption)
     
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (batch_size, nq, 256)

        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (batch_size, nq, 4)
        all_detections = []
        tokenizer = self.grounding_model.model.tokenizer
        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # (n, 256)
        boxes = prediction_boxes[mask]  # (n, 4)
        tokenized = tokenizer(caption)
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]
        logits = logits.max(dim=1)[0]
        source_h, source_w, _ = image.shape
        detections = self.grounding_model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits
        )
        class_id = self.grounding_model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        all_detections.append(detections)
        return all_detections
    

    
    def predict_laser(self, 
                      image, 
                      boxes, 
                      target_names,
                      target_relation_kws,
                      target_relation_preds,
                      query_names,
                      query_relations,
                      query_relation_kws,
                      query_attributes,
                      ):
        
        # Preprocess query attributes for the case where one object having multiple attribute
        query_attribute_ls = [[attr, f"not {attr}"] for attr in query_attributes]
        prob_labels, selected_object_bboxes = [], []
        
        if not len(boxes) == 0:
            height, width = image.shape[0], image.shape[1]
            batched_pred_masks = [bbox_to_mask(bbox, height, width) for bbox in boxes]
            batched_object_ids = [[0, 0, i] for i in range(len(batched_pred_masks))]
            
            batched_image_cate_probs, _, _, _ = \
                self.predicate_model(
                    batched_video_ids=[0],
                    batched_videos=[image],
                    batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
                    batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
                    batched_names=[query_names],  # Dataset-wise categorical labels
                    batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
                    batched_unary_kws=[[]],  # Dataset-wise unary predicate labels
                    batched_binary_kws=[[]],  # Dataset-wise binary predicate labels
                    batched_obj_pairs=[],  # Ground truth binary relations
                    batched_video_splits=[0],  # [number of videos]
                    batched_binary_predicates=[[]],  # None indicates inference time
                )
                    
            all_unary_res = []
            for unary in query_attribute_ls:
                _, batched_image_unary_probs, _, _ = \
                    self.predicate_model(
                        batched_video_ids=[0],
                        batched_videos=[image],
                        batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
                        batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
                        batched_names=[query_names],  # Dataset-wise categorical labels
                        batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
                        batched_unary_kws=[unary],  # Dataset-wise unary predicate labels
                        batched_binary_kws=[[]],  # Dataset-wise binary predicate labels
                        batched_obj_pairs=[],  # Ground truth binary relations
                        batched_video_splits=[0],  # [number of videos]
                        batched_binary_predicates=[None],  # None indicates inference time
                    )
                all_unary_res.append(batched_image_unary_probs)
            
            # Combine the uanry and object name information together 
            # Step 1: reformat the object name, attributes, and relations into better format
            predicted_name = {}
            for (oid, name) in batched_image_cate_probs[0]:
                prob = batched_image_cate_probs[0][(oid, name)].item()
                if not oid in predicted_name:
                    predicted_name[oid] = []
                predicted_name[oid].append((prob, name))
            
            for oid in predicted_name:
                predicted_name[oid] = sorted(predicted_name[oid], reverse=True)   
                
            reformatted_unary_res = {}
            for unary_res in all_unary_res:
                for  (fid, oid, kw) in unary_res[0].keys():
                    if not oid in reformatted_unary_res:
                        reformatted_unary_res[oid] = {}
                    if kw in query_attributes:
                        reformatted_unary_res[oid][kw] = unary_res[0][(fid, oid, kw)].item()
            
            topk_obj_pair_info = get_topk_per_relation(predicted_name, binary_predicates=query_relations, top_k=30)
            top_obj_pairs = merge_object_pair_ids(topk_obj_pair_info)

            batched_image_rel_probs = {}
            if len(top_obj_pairs) > 0:
                _, _, batched_image_rel_probs, _ = \
                    self.predicate_model(
                        batched_video_ids=[0],
                        batched_videos=[image],
                        batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
                        batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
                        batched_names=[query_names],  # Dataset-wise categorical labels
                        batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
                        batched_unary_kws=[[]],  # Dataset-wise unary predicate labels
                        batched_binary_kws=[query_relation_kws],  # Dataset-wise binary predicate labels
                        batched_obj_pairs=top_obj_pairs,  # Ground truth binary relations
                        batched_video_splits=[0],  # [number of videos]
                        batched_binary_predicates=[query_relations],  # None indicates inference time
                    )
                
            # Step 3: Aggregate the object name and attribute predictions together
            aggr_res = {}            
            
            for oid in predicted_name:
                top_prob, name = predicted_name[oid][0]
                if not oid in aggr_res:
                    aggr_res[oid] = {}
                
                if name in target_names:
                    aggr_res[oid]['name'] = top_prob, name
                else:
                    # the top predicted name is not the target name
                    aggr_res[oid]['name'] = 0, "other"
                    
            for oid in reformatted_unary_res: 
                
                if not oid in aggr_res:
                    continue
                
                if not 'attr' in aggr_res[oid]:
                    aggr_res[oid]['attr'] = []
                for attr in reformatted_unary_res[oid]:
                    aggr_res[oid]['attr'].append((reformatted_unary_res[oid][attr], attr))
            
            if len(batched_image_rel_probs) > 0:
                rel_preds = batched_image_rel_probs[0]
            else:
                rel_preds = {}

            identified_rels = []
            for (_, (subj, obj), rel), p in rel_preds.items():
                subj_prob, pred_subj_name = predicted_name[subj][0]
                obj_prob, pred_obj_name = predicted_name[obj][0]
                rel_prob = p
                if (rel, pred_subj_name, pred_obj_name) in target_relation_preds:
                    prob = subj_prob * obj_prob * rel_prob
                    if prob > self.rel_thres:
                        identified_rels.append((prob, (rel, subj, obj), (rel, pred_subj_name, pred_obj_name)))
                
            # Thresholding
            avg_aggr_res = {}
            for oid in aggr_res:
                name_prob, name = aggr_res[oid]['name']
                
                # Target Name must rank at top 1
                if name_prob == 0:
                    continue
                
                # Attribute aggregation
                probs = [name_prob]

                if 'attr' in aggr_res[oid]:
                    probs += [p for p, _ in aggr_res[oid]['attr']]
                
                avg_val = sum(probs) / len(probs)
                    
                if avg_val > self.aggr_thres:
                    avg_aggr_res[oid] = avg_val
            
            # Relation aggregation
                
            # Only pick the top-k object according to the setip
            top_objects_probs = get_topk_per_object_name(aggr_res, avg_aggr_res, k=self.k)
            # top_objects_probs = {oid: prob for prob, oid in top_objects}
            
            selected_objects = list(top_objects_probs.keys())
            selected_object_bboxes = boxes[selected_objects]
                                
            prob_labels = []
            for oid in top_objects_probs:
                prob, label = predicted_name[oid][0]
                prob_labels.append(f"{prob:.2f}::{label}")

        return prob_labels, selected_object_bboxes, identified_rels
    
        
    def act_custom(self, prompt, obs):
        assert type(obs) == str, "input image path is in incorrect type" # input image path
        out = self.model.respond(prompt, obs)
        # fix common generated json errors
        out = out.replace("'",'"') 
        out = out.replace('\"s ', "\'s ")
        out = out.replace('\"re ', "\'re ")
        out = out.replace('\"ll ', "\'ll ")
        out = out.replace('\"t ', "\'t ")
        out = out.replace('\"d ', "\'d ")
        out = out.replace('\"m ', "\'m ")
        out = out.replace('\"ve ', "\'ve ")
        out = out.replace('```json', '').replace('```', '')
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action = self.json_to_action(out)
        self.planner_steps += 1
        return action, out

    def act(self, observation, user_instruction):
        
        if type(observation) == dict:
            obs = copy.deepcopy(observation[self.obs_key])
        else:
            # assert len(observation) == 1
            obs = copy.deepcopy(observation) # input image path

        
        prompt, scene_graph_prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        # some models do not support json scheme, add style into prompt
        if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or self.model_type == 'custom':
            prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == 'custom':
            return self.act_custom(prompt, obs) 
        
        sg_message = self.get_message(obs, scene_graph_prompt)
        out = ''
        try:
            out = self.model.respond(sg_message, get_scene_graph=False, get_entity_name=True)
        except Exception as e:
            print(e)
            if 'qwen' in self.model_name:
                return -2, '''{
                    "visual_state_description": "qwen model generate empty action due to inappropriate content check",
                    "reasoning_and_reflection": "invalid json, random action",
                    "language_plan": "invalid json, random action"
                }'''

        try:
            sg_response = json.loads(out)
        except:
            sg_response = []

        print("Scene graph message:", sg_message[-1]['content'][-1]['text'])
        print("Scene graph response:", sg_response)

        if 'scene_graph_info'  in sg_response and len(sg_response['scene_graph_info']) > 0 and self.sg_type != "None":

            target_names, current_names, query_objects, \
            target_relation_preds, current_relation_preds, \
                target_relation_kws, query_relation_kws, query_relation_preds = self.get_sg_kws(sg_response['scene_graph_info'])

            if type(obs) == str:
                obs = [obs]
            if type(observation) == str:
                observation = [observation]

            for ct, image_path in enumerate(observation):
                image_path_ls = image_path.split('/')
                image_dir, image_name = image_path_ls[:-1], image_path_ls[-1]
                image_name = image_name.split('.')[0]
                new_image_name = image_name + '_bbox.png'
                new_image_path = os.path.join(*image_dir, new_image_name)

                if os.path.exists(new_image_path):
                    obs = [new_image_path]
                    continue

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                height, width = image.shape[0], image.shape[1]
                current_names_w_object = current_names
                detection_ls = self.predict_gd(image, current_names_w_object, box_threshold=0.15, text_threshold=0.1)
                assert len(detection_ls) == 1, "Detection ls is not equal to 1!!"
                detections = detection_ls[0]

                if detection_ls[0].xyxy.shape[0] == 0:
                    print("No boxes detected")
                    break

                scene_graph_text = []
                reformated = torch.tensor([
                    (
                        get_valid_bbox_val(x1, width), 
                        get_valid_bbox_val(y1, height),
                        get_valid_bbox_val(x2, width), 
                        get_valid_bbox_val(y2, height)
                    )
                    for x1, y1, x2, y2 in detections.xyxy
                ])

                boxes = []
                labels = []

                for ct, class_id in enumerate(detections.class_id):
                    if class_id is None:
                        continue
                    boxes.append(reformated[ct])
                    labels.append(current_names_w_object[class_id])

                if len(boxes) == 0:
                    boxes = torch.tensor([])
                else:
                    boxes = torch.stack(boxes)

                if self.sg_type == "GD":
                    save_prompts_one_image(
                        frame_image=image,
                        boxes=boxes,
                        labels=labels,
                        points=[],
                        save_path=new_image_path
                    )
                elif self.sg_type == "LASER":
                    save_prompts_one_image(
                        frame_image=image,
                        boxes=boxes,
                        labels=labels,
                        points=[],
                        save_path=new_image_path.replace("bbox", "bbox_gd")
                    )
                    if len(boxes) != 0:
                        target_labels, target_bboxes, identified_rels = self.predict_laser(
                            image=image, 
                            boxes=boxes.numpy(), 
                            target_names=target_names,
                            target_relation_kws=target_relation_kws,
                            target_relation_preds=target_relation_preds,
                            query_names=list(set(query_objects)),
                            query_attributes=[],
                            query_relation_kws=query_relation_kws,
                            query_relations=query_relation_preds,
                        )
                    else:
                        target_labels = []
                        target_bboxes = []

                    if not len(target_bboxes) == 0:
                        save_prompts_one_image(frame_image=image, 
                                                boxes=torch.Tensor(target_bboxes), 
                                                labels=["" for _ in target_bboxes], 
                                                points=[], 
                                                save_path=new_image_path)
                        save_prompts_one_image(frame_image=image, 
                                                boxes=torch.Tensor(target_bboxes), 
                                                labels=target_labels, 
                                                points=[], 
                                                save_path=new_image_path.replace("bbox", "bbox_debug"))

                        obs = [new_image_path]
                        image_path = None

                    if not len(target_labels) == 0:
                        prompt += f" Here is a list of objects detected in the bounding box at frame idx {ct}, along with their probabilities:  {target_labels}."

                if image_path and not image_path in obs:
                    obs = [image_path]

            assert len(obs) <= 1, "Observation is more than one!"
            print("Finished extracting objects from image")

        if len(self.episode_messages) == 0:
             self.episode_messages  = self.get_message(obs, prompt, scene_graph=sg_response['scene_graph_info'] if 'scene_graph_info' in sg_response and len(sg_response['scene_graph_info']) > 0 else None)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages, scene_graph=sg_response['scene_graph_info'] if 'scene_graph_info' in sg_response and len(sg_response['scene_graph_info']) > 0 else None)
            else:
                self.episode_messages = self.get_message(obs, prompt, scene_graph=sg_response['scene_graph_info'] if 'scene_graph_info' in sg_response and len(sg_response['scene_graph_info']) > 0 else None)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")
                    print("Lang plan prompt:", text_content)

        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try: 
                out = self.model.respond(self.episode_messages)
                time.sleep(15)
            except Exception as e:
                print("An unexpected error occurred:", e)
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            try: 
                out = self.model.respond(self.episode_messages)
            except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        logger.debug(f"Model Output:\n{out}\n")
        print("Model Output__:", out)
        self.sg_response = json.dumps(sg_response)
        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
        action = self.json_to_action(out)
        self.planner_steps += 1
        return action, out

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])
