import torch
import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url, template, template_lang, fix_json
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger


###################
# SAM2
import sys
sys.path.append("/workspace/jayj/EB-LASER/sam2")
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import cv2
import argparse
import hydra
import matplotlib.pyplot as plt
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path="../../../sam2/sam2/configs/sam2.1") # WHY MUST BE RELATIVE?!

def show_mask(mask, ax, obj_id=None, det_class=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        cmap = plt.get_cmap("gist_rainbow")
        cmap_idx = 0 if obj_id is None else obj_id
        color = list(cmap((cmap_idx * 47) % 256))
        color[3] = 0.5
        color = np.array(color)
        
    mask_image = mask * color.reshape(1, 1, -1)

    # draw a box around the mask with the det_class as the label
    if not det_class is None:
        
        # Find the bounding box coordinates
        y_indices, x_indices, _ = np.where(mask > 0)
        if y_indices.size > 0 and x_indices.size > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            rect = plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1.5,
                edgecolor=color[:3],
                facecolor="none",
                alpha=color[3]
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 5,
                f"{det_class}",
                color="white",
                fontsize=6,
                backgroundcolor=np.array(color),
                alpha=1
            )
            
    ax.imshow(mask_image)

def save_mask_one_image(frame_image, masks, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis('off')

    if type(masks) == list:
        masks = {i: mask for i, mask in enumerate(masks)}
        
    # Add the bounding boxes
    for obj_id, mask in masks.items():
        show_mask(mask, ax, obj_id=obj_id, det_class=None, random_color=False)

    # Show the plot
    plt.savefig(save_path)
    plt.close()


def save_mask_contours_one_image(frame_image, masks, save_path):
    if type(masks) == list:
        masks = {i: mask for i, mask in enumerate(masks)}
    contour_img = frame_image.copy()

    # Loop through all masks and draw their contours
    for obj_id, mask in masks.items():

        segmentation = mask.astype(np.uint8)
        contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, color=(0, 255, 0), thickness=2)

    # Save the result
    cv2.imwrite(save_path, cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))


###################

###################
## Grounding DINO

import sys
sys.path.append("/workspace/jayj/EB-LASER/GroundingDINO")
from groundingdino.util.inference import Model as gd_Model
from groundingdino.util.inference import load_image as load_gd_image
from groundingdino.util.utils import get_phrases_from_posmap
import torch
import math
from torch.cuda.amp import autocast

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
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))
    
    if label is not None:
        ax.text(x0, y0 - 5, label, color='white', fontsize=10, backgroundcolor=np.array(color[:3]), alpha=color[3])
    
    
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
    
    for object_id, (point_ls, label_ls) in enumerate(zip(points, labels)):
        if not len(point_ls) == 0:
            show_points(point_ls.cpu(), label_ls.cpu(), ax, object_id=object_id)
        
    # Show the plot
    plt.savefig(save_path)
    plt.close()


def save_mask_contours_and_boxes_one_image(frame_image, masks, boxes, save_path):
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
    for box in boxes:
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(contour_img, (x0, y0), (x1, y1), color=(255, 0, 0), thickness=2)

    # Save the result
    cv2.imwrite(save_path, cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))

###################

class VLMPlanner():
    def __init__(self, model_name, model_type, actions, system_prompt, examples, n_shot=0, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1, 
                scene_graph_examples = '', sam2_config_path = '', sam2_checkpoint_path = '', 
                gd_config_path='', gd_checkpoint_path='', kwargs={}):

        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.set_actions(actions)
        self.model_type = model_type
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        #####################
        # SAM2
        sam2 = build_sam2(sam2_config_path, sam2_checkpoint_path, device="cuda:0", apply_postprocessing=False)
        sam_target_fps = 1
        prompt_only = True

        self.predictor = SAM2ImagePredictor(sam2)

        # self.mask_generator = SAM2AutomaticMaskGenerator(
        #     model=sam2,
        #     points_per_side=32,
        #     points_per_batch=32,
        #     pred_iou_thresh=0.8,
        #     stability_score_thresh=0.8,
        #     crop_n_layers=2,
        #     box_nms_thresh=0.8,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=1000.0,
        #     use_m2m=True,
        # )

        #####################

        #####################
        ## Grounding DINO
        self.scene_graph_examples = scene_graph_examples[:n_shot]

        self.scene_graph_prompt = f'''## You are a robot operating in a home. You are given an image and observe the environment for performing the given task. \
            The input given to you is {'a first person view observation'}\
            You are supposed to output in JSON.'''
        self.grounding_model = gd_Model(
            model_config_path=gd_config_path, 
            model_checkpoint_path=gd_checkpoint_path
        )

        self.grounding_model.model = self.grounding_model.model.to("cuda:0")
        #####################

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')

    
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


    def process_prompt(self, user_instruction, prev_act_feedback=[]):
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            scene_graph_prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
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
            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            
            scene_graph_prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

        return prompt, scene_graph_prompt
    

    def get_message(self, image, prompt, messages=[]):
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
            else:
                # if isinstance(image, str)
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                print("Image:", image)
                cv2.imwrite(image_path, image)

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

    
        
    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
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

    def predict_sam2(self, image, classes):
        """
        Dealing with a single image
        1. Preprocess "classes" to be able to fed into SAM2 as prompt
        2. receive the outputs from SAM2

        Identify image data type

        """
        pass

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

    def act(self, observation, images, user_instruction):
        height, width = images[0].shape[0], images[0].shape[1]
        
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path

        prompt, scene_graph_prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        # some models do not support json scheme, add style into prompt
        if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or self.model_type == 'custom':
            prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == 'custom':
            return self.act_custom(prompt, obs) 

        ####################################
        # GD + SAM2

        sg_message = self.get_message(obs, scene_graph_prompt)
        try:
            out = self.model.respond(sg_message, get_scene_graph=True)
        except Exception as e:

            print(e)
            if 'qwen' in self.model_name:
                return -2,'''{"visual_state_description":"qwen model generate empty action due to inappropriate content check", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
        
        sg_keywords = json.loads(out)['scene_graph_entities']
        print(sg_keywords)

        obs = []

        prompt = prompt + f'\n\n The current image height is {height}, width is {width}.'
        # breakpoint()
        for ct, image in enumerate(images):
            ####################################
            # Grounding DINO

            with autocast(dtype=torch.float32):  # instead of bfloat16
                detections = self.predict_gd(image, sg_keywords, box_threshold=0.25, text_threshold=0.4)
            reformated = torch.tensor([(get_valid_bbox_val(x1, width), 
                get_valid_bbox_val(y1, height),
                get_valid_bbox_val(x2, width), 
                get_valid_bbox_val(y2, height))
                for x1, y1, x2, y2 in detections[0].xyxy])
            ####################################

            ####################################
            # SAM2
            # masks = self.mask_generator.generate(image.copy())
            self.predictor.set_image(np.array(image)) 
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=reformated,
                multimask_output=False,
            )

            # masks = [m['segmentation'].reshape(height, width, 1) for m in masks]
            masks = [masks[i].squeeze(0) for i in range(len(masks))]
            # breakpoint()
            image_path = observation[-1].split('/') if isinstance(observation, list) else observation.split('/')
            image_dir, image_name = image_path[:-1], image_path[-1]
            image_name = image_name.split('.')[0]
            new_image_name = image_name + '_bbox.png'
            new_image_path = os.path.join(*image_dir, new_image_name)
            breakpoint()
            # save_mask_one_image(frame_image=image, masks=masks, save_path=new_image_path)
            save_mask_contours_and_boxes_one_image(frame_image=image, masks=masks, save_path=new_image_path)
            breakpoint()
            obs.append(new_image_path)
            scene_graph_text += [f"{obj_name}: {obj_box}" for (obj_name, obj_box) in zip(sg_keywords, reformated)]
            prompt += f"Its scene graph at frame {ct} is:  {scene_graph_text}."

        if isinstance(obs, list) and len(set(obs)) == 1:
            obs = obs[0]
        ####################################
        print("Finished extracting dino+sam2 results")

        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

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
