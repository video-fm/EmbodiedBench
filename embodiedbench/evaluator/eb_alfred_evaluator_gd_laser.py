import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["remote_url"] = "http://0.0.0.0:15235/v1"
os.environ["DISPLAY"] = "cherry:11.0"
os.environ["OPENAI_API_KEY"] = "sk-proj-klfjalfadsil;fkadsl;kfsjfkl"

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


scene_graph_examples = [
'''Human Instruction: Navigate to the floorlamp and be as close as possible.
   
   Visual Description: I can see the floorlmap is at left side in front of me, next to the TVStand. Around me is a Cabinet at my left side.
   
   Scene Graph Entity in JSON: "scene_graph_entities": ["floorlmap", "TVStand", "cabinet"]
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



laser_sg_examples_no_revert = [
'''
Human Instruction: Place two yellow bottles on top of a toilet.

Candidate Target Entities location:
image 0: [["SoapBottle", "color: yellow", "bbox: [120, 220, 160, 280]"], ["SprayBottle", "color: yellow", "bbox: [200, 250, 240, 300]"], ["Toilet", "bbox: [150, 300, 300, 400]"]]

Reasoning and reflection: Based on the visual description, I can see a SoapBottle and a SprayBottle that are both yellow in color. These match the target instruction to place "two yellow bottles" on a Toilet. The toilet is visible in the scene as well. To complete the task, I will pick up the yellow SoapBottle first.

Executable plans: [[76, find a SoapBottle], [126, pick up the SoapBottle]]

Environment Feedback: Step 1 is executed successfully; Step 2 is executed successfully.

Candidate Target Entities location:
image 0: [["SprayBottle", "color: yellow", "bbox: [200, 250, 240, 300]"], ["Toilet", "bbox: [150, 300, 300, 400]"]]

Reasoning and reflection: Now that I am holding the SoapBottle, I attempted to pick up the SprayBottle as the second yellow bottle. However, the action was invalid since I cannot hold two objects simultaneously. Therefore, I will navigate to the Toilet and place the SoapBottle down first.

Executable plans: [[17, find a Toilet], [133, put down the object in hand]]

Environment Feedback: Step 3 is executed successfully; Step 4 is executed successfully.

Candidate Target Entities location:
image 0: [["SprayBottle", "color: yellow", "bbox: [200, 250, 240, 300]"], ["Toilet", "bbox: [150, 300, 300, 400]"]]

Reasoning and reflection: The SoapBottle is now on the Toilet. I will proceed to pick up the SprayBottle and place it on the Toilet as well to satisfy the instruction.

Executable plans: [[10, find a SprayBottle], [90, pick up the SprayBottle], [17, find a Toilet], [133, put down the object in hand]]

Environment Feedback: Step 5 is executed successfully; Step 6 is executed successfully; Step 7 is executed successfully; Step 8 is executed successfully.

Candidate Target Entities location:
image 0: [["SoapBottle", "on top of Toilet"], ["SprayBottle", "on top of Toilet"], ["Toilet"]]

Reasoning and reflection: Both yellow bottles—the SoapBottle and the SprayBottle—have been successfully placed on the Toilet. This satisfies the original instruction.

Executable plans: []

Task Success
'''
,
'''Human Instruction: Hold a pillow while turning on a lamp.

Candidate Target Entities location:  
image 0: [["Pillow", "color: blue", "bbox: [180, 240, 220, 300]"], ["FloorLamp", "bbox: [300, 280, 360, 400]"], ["CoffeeTable", "bbox: [150, 300, 300, 380]"], ["CreditCard", "bbox: [160, 310, 180, 320]"], ["Plate", "bbox: [210, 320, 250, 340]"], ["RemoteControl", "bbox: [180, 315, 200, 330]"]]

Reasoning and reflection: I can see a Pillow on the coffee table—it is blue in color and easily reachable. The FloorLamp is visible in the background. According to the instruction, I need to be holding the pillow while I turn on the lamp. I will first pick up the pillow.

Executable plans: [[74, find a Pillow], [124, pick up the Pillow]]

Environment Feedback: Step 1 is executed successfully; Step 2 is executed successfully.

Candidate Target Entities location:  
image 0: [["FloorLamp", "bbox: [300, 280, 360, 400]"], ["TVStand", "bbox: [100, 250, 200, 350]"], ["Television", "bbox: [120, 200, 180, 260]"]]

Reasoning and reflection: I am currently holding the pillow. The FloorLamp is clearly visible in front of me. The next step is to walk to the lamp and turn it on. Since the path appears clear, I can directly interact with it.

Executable plans: [[27, find a FloorLamp], [153, turn on the FloorLamp]]

Environment Feedback: Step 3 is executed successfully; Step 4 is executed successfully.

Candidate Target Entities location:
image 0: [["Pillow", "in hand"], ["FloorLamp", "switched on"]]

Reasoning and reflection: The task required me to hold a pillow while turning on a lamp. I successfully picked up the pillow and activated the FloorLamp. All steps were executed correctly, and I am still holding the pillow.

Executable plans: []

Task Success

'''
,
'''Human Instruction: As you prepare for dinner, arrange an eating surface that will hold your food with a spoon in it on the kitchen table.

Candidate Target Entities location:  
image 0: [["DiningTable", "bbox: [300, 200, 400, 300]"], ["Spoon", "bbox: [150, 250, 180, 270]"], ["Plate", "bbox: [220, 240, 260, 270]", "parent: CounterTop"], ["CounterTop", "bbox: [200, 220, 300, 270]"], ["Sink", "bbox: [210, 225, 250, 260]"]]

Reasoning and reflection: I need to construct an eating surface by placing a spoon into a plate and then moving that plate onto the dining table. I will begin by picking up the spoon since it is directly visible and not embedded inside another receptacle.

Executable plans: [[105, pick up the Spoon]]

Environment Feedback: Step 1 is executed successfully.

Candidate Target Entities location:  
image 0: [["Plate", "bbox: [220, 240, 260, 270]", "parent: CounterTop"]]

Reasoning and reflection: I can see that the plate is on the countertop. However, I am already holding the spoon. I will attempt to pick up the plate directly and hope the action succeeds.

Executable plans: [[122, pick up the Plate]]

Environment Feedback: Step 2 is invalid. Plate is not visible because it is in CounterTop. Go there to pick the object instead.

Candidate Target Entities location:  
image 0: [["CounterTop", "bbox: [200, 220, 300, 270]"]]

Reasoning and reflection: The error indicates the plate cannot be picked up unless I move to the CounterTop first. I will move to the CounterTop.

Executable plans: [[39, find a CounterTop]]

Environment Feedback: Step 3 is executed successfully.

Candidate Target Entities location:  
image 0: [["Plate", "bbox: [220, 240, 260, 270]"], ["Spoon", "in hand"]]

Reasoning and reflection: I am now at the CounterTop. However, I cannot pick up the plate while holding the spoon. I will put the spoon down first.

Executable plans: [[133, put down the object in hand]]

Environment Feedback: Step 4 is executed successfully.

Candidate Target Entities location:  
image 0: [["Plate", "bbox: [220, 240, 260, 270]"]]

Reasoning and reflection: With my hand now free, I can attempt to pick up the plate.

Executable plans: [[122, pick up the Plate]]

Environment Feedback: Step 5 is executed successfully.

Candidate Target Entities location:  
image 0: [["DiningTable", "bbox: [300, 200, 400, 300]"], ["Plate", "in hand"]]

Reasoning and reflection: I now need to place the plate on the dining table to complete the eating surface setup.

Executable plans: [[18, find a DiningTable], [133, put down the object in hand]]

Environment Feedback: Step 6 is executed successfully; Step 7 is executed successfully.

Candidate Target Entities location:  
image 0: [["Spoon", "bbox: [150, 250, 180, 270]"], ["Plate", "on DiningTable"]]

Reasoning and reflection: The plate has been placed on the dining table. I now need to re-locate and pick up the spoon again so I can place it into the plate.

Executable plans: [[19, find a Spoon], [105, pick up the Spoon]]

Environment Feedback: Step 8 is executed successfully; Step 9 is executed successfully.

Candidate Target Entities location:  
image 0: [["Spoon", "in hand"], ["Plate", "on DiningTable"]]

Reasoning and reflection: I am now holding the spoon. The plate is on the dining table. I will now place the spoon into the plate to complete the task.

Executable plans: [[18, find a DiningTable], [133, put down the object in hand]]

Environment Feedback: Step 10 is executed successfully; Step 11 is executed successfully.

Candidate Target Entities location:  
image 0: [["Spoon", "in Plate", "on DiningTable"]]

Reasoning and reflection: The spoon has been placed inside the plate, which is on the dining table, successfully setting up the eating surface for dinner.

Executable plans: []

Task Success

'''
] 


#common sense, base, and ...
target_examples = [
'''
   Environmental Feedback: The last action succeeded. 
    
   Human Instruction:  Place two yellow bottles on top of a toilet
      
   Last Visual Description: I am in a bathroom. I can see a toilet in front of me. I need to find two yellow bottles (SoapBottles) and place them on the toilet..
   
   Target Entities in JSON: {"target_name": ["SoapBottle", "Toilet"], "target_attribute": ["in front"], "target_relation": [["SoapBottles", "on top", "Toilet"]], "related_objects": ['Toilet', 'ToiletPaper', 'ToiletPaperRoll', 'ToiletPaperHanger', 'Plunger', 'HandTowel', 'HandTowelHolder', 'SoapBar', 'SoapBottle', 'Sink', 'SinkBasin', 'Towel', 'TowelHolder', 'Mirror', 'ShowerDoor', 'ShowerGlass', 'Bathtub', 'BathtubBasin']
}
''',

''' 
    Environmental Feedback: Last action is invalid. Ladle is not visible because it is in DiningTable. Go there to pick the object instead.

    Human Instruction: Rinse off a ladle and move it to the table.

    Last Visual Description: I am in a room with a garbage can and a fridge. The floor is tiled. There is a door in the background.

    Target Entities in JSON: {
    "target_name": ["Ladle", "Sink", "DiningTable"],
    "target_attribute": ["tiled floor", "in the background"],
    "target_relation": [["Ladle", "in", "DiningTable"], ["Ladle", "rinsed in", "Sink"], ["Ladle", "moved to", "DiningTable"]],
    "related_objects": [
        "GarbageCan", "Fridge", "Ladle", "Sink", "SinkBasin", "DiningTable", "Cup", "Plate", "Chair", "CounterTop", "Cabinet"
    ]
    }
''', 

''' 
Environmental Feedback: Last action executed successfully.

Human Instruction: Rinse off something for serving soup and move it to the table.

Last Visual Description: I can see a bread, an egg, a ladle and a dining table.

Target Entities in JSON: {
  "target_name": ["Ladle", "Sink", "DiningTable"],
  "target_attribute": [],
  "target_relation": [["Ladle", "rinsed in", "Sink"], ["Ladle", "moved to", "DiningTable"]],
  "related_objects": [
    "Ladle", "DiningTable", "Bread", "Egg", "Sink", "SinkBasin", "Cup", "Plate", "Chair", "CounterTop", "Cabinet"
  ]
''', 
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
            
            if self.config['sg_text']:
                few_shot_examples = laser_sg_examples_no_revert
            else:
                few_shot_examples = examples
            examples = json.load(open(example_path, 'r+')) if self.eval_set != 'long_horizon' else json.load(open(exploration_example_path, 'r+'))
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
        parser.add_argument('--exp_name', default='history_laser_bbox_0.1_with_bbox_with_text', type=str, help='Name of the experiment.')
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
        'n_shots': 3,
        'down_sample_ratio': 1.0,
        'model_type': 'remote', # 'local', 
        'language_only': 0,
        'exp_name': 'laser_alfred_try1',
        'chat_history': 0, 
        'detection_box': True,
        'multiview': False, 
        # 'eval_sets': ['base', 'spatial', 'visual_appearance', 'common_sense', 'complex_instruction', 'long_horizon'], 
        'eval_sets': ['long_horizon'],
        'selected_indexes': list(range(6,50)), 
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
        'top_k': 1, 
        'aggr_thres': 0.3,
        'gd_config_path': '/home/asethi04/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        'gd_checkpoint_path': '/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth'
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    evaluator = EB_AlfredEvaluator(config)
    evaluator.evaluate_main()




