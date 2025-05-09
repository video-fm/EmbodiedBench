import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["remote_url"] = "http://0.0.0.0:15235/v1"
os.environ["DISPLAY"] = "cherry:11.0"

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

Human Instruction : Navigate to the floorlamp and be as close as possible.

Candidate Target Entities location: 
image 0: [] 

Reasoning and reflection: I cannot see the floorlamp in my current field of view.
So I will rotate my view ninety degrees to the right only once to find the floorlamp. \
Executable plans: [[0, Rotate to the right by 90 degrees]]

Environment Feedback: Step 1 is executed successfully;

Human Instruction : Navigate to the floorlamp and be as close as possible.

Candidate Target Entities location: 
image 0: []
image 1: [[0, 75, 25, 150], [300, 320, 200, 240]] 

Reasoning and reflection: I can see the floorlmap is at left side in front of me, next to the TVStand. \
This floorlamp is corresponding to the location [0, 75, 25, 150]. \
There could be another floor lamp at [300, 320, 200, 240], but it is more far away from me so I will visit [0, 75, 25, 150] first. \
Around me is a Cabinet at my left side. Following the guideline, since the floorlamp is at left side, I can approach it\
by a series of moving forward and left/rightward. Since currently there is an obstacle at my left, I will first move forward twice.
Executable plans: [[0, Move forward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 1 is executed successfully; Step 2 is executed successfully;

Candidate Target Entities location: 
image 0: [] 
image 1: [[0, 75, 25, 150], [300, 320, 200, 240]] 
image 2: [[0, 95, 15, 170], [320, 320, 100, 140]] 

Reasoning and reflection: I can see now I get closer to the floorlamp. 
This floorlamp is corresponding to the location[0, 95, 15, 170]. \
There could be another floor lamp at [300, 320, 200, 240], but it is more far away from me so I will visit [0, 95, 15, 170] first. \
My target floorlamp is still located in front of me at the left. I will try move forward and move leftward in series.
Executable plans: [[0, Move forward by 0.25], [3, Move leftward by 0.25], [0, Move forward by 0.25], [3, Move leftward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 3 is executed successfully; Step 4 is executed successfully; Step 5 is executed successfully; Step 6 is executed successfully; Step 7 is Invalid, as a StandardIslandHeight is blocking you from moving forward.

Candidate Target Entities location: 
image 0: [[0, 75, 25, 150], [300, 320, 200, 240]] 
image 1: [[0, 95, 15, 170], [320, 320, 100, 140]] 
image 2: [[0, 120, 10, 200], [340, 350, 120, 150]] 

Reasoning and reflection: I am a lot closer to the floorlamp by a series a forward and leftward movement. \
This floorlamp is corresponding to the location [0, 120, 10, 200]. \
There could be another floor lamp at [340, 350, 120, 150] but it is more far away from me so I will visit [0, 120, 10, 200] first. \
It is still in front of me. However, The last action is invalid, showing that I am blocked from moving forward.\
That means there is an obstacle in front of me and I can see it is probably a small table. Following the strategy, I can get away by first moving to the left and move forward. 
Executable plans: [[3, Move leftward by 0.25], [3, Move leftward by 0.25], [0, Move forward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 8 is executed successfully; Step 9 is executed successfully; Step 10 is executed successfully; Step 11 is executed successfully;

Candidate Target Entities location: 
image 0: [[0, 95, 15, 170], [320, 320, 100, 140]] 
image 1: [[0, 120, 10, 200], [340, 350, 120, 150]] 
image 2: [[40, 120, 80, 300,], [380, 350, 160, 150]] 

Reasoning and reflection: I can see that the floorlamp is at the right side of me, slight in the front. \
This floorlamp is corresponding to the location [40, 120, 80, 300,]. \
There could be another floor lamp at [380, 350, 160, 150] but it is more far away from me so I will visit [40, 120, 80, 300,] first. \
I think it in effect of consecutively moving leftward. Also, I have got away with the obstacle.\
Now I need to move right and I will probably be close to the floorlamp.
Executable plans: [[2, Move rightward by 0.25], [2, Move rightward by 0.25]]]

Environment Feedback: Step 12 is executed successfully; Step 13 is executed successfully. 

Candidate Target Entities location: 
image 0: [[0, 120, 10, 200], [340, 350, 120, 150]]
image 1: [[40, 120, 80, 300,], [380, 350, 160, 150]]
image 2: [[20, 120, 60, 300,], [360, 350, 140, 150]] 

Reasoning and reflection: Last two action brought me closer to the floorlamp. \
This floorlamp is corresponding to the location [20, 120, 60, 300,]. \
There could be another floor lamp at [360, 350, 140, 150] but it is more far away from me so I will visit [20, 120, 60, 300,] first. \
following the strategy, I should not complete now since I can still be closer to the floorlamp by moving forward a few steps. 
Executable plans: [[0, Move forward by 0.25], [0, Move forward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 14 is executed successfully; Step 15 is executed successfully; Step 16 is executed successfully.


Task Success
'''
,
'''
Human Instruction: Navigate to the computer on the desk and be as close as possible.

Candidate Target Entities location: 
image 0: [[35, 75, 65, 150], [300, 75, 400, 150]]  

Reasoning and reflection: According to your instruction, I first need to locate the target object. \
The candidate computer is at [35, 75, 65, 150]. 
There could be another computer at [300, 75, 400, 150], but it is more far away from me so I will visit [35, 75, 65, 150] first. \
I can see that the computer is located in my left front direction, placed on a desk. Next, I analyze the obstacles around me. In front of me, slightly to the right, there is a chair blocking my direct path. To avoid it, I decide to move leftward first, followed by a forward movement to approach the desk.
Executable plans: [[0, Move forward by 0.25], [3, Move leftward by 0.25]]

Environment Feedback: Step 1 is executed successfully; Step 2 is Invalid, as a StandardHeightTable is blocking you from moving forward.

Candidate Target Entities location: 
image 0: [[35, 75, 65, 150], [300, 75, 400, 150]]
image 1: [[25, 45, 55, 160], [320, 55, 420, 180]] 

Reasoning and reflection: I am now closer to the desk, and I can still see the computer on it. \
The candidate computer is at [25, 45, 55, 160]. \
There could be another computer at [320, 55, 420, 180], but it is more far away from me so I will visit [25, 45, 55, 160] first. \
It remains in my left front direction. My last leftward movement failed because a StandardHeightTable is blocking my path. However, following the strategy, \
I need not address the invalid action since forward movement can still bring me closer to the target object. So I will still make forward movements for now. 
Executable plans: [[0, Move forward by 0.25], [0, Move forward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 3 is executed successfully; Step 4 is executed successfully; Step 5 is Invalid, as a StandardHeightTable is blocking you from moving forward.

Candidate Target Entities location: 
image 0: [[35, 75, 65, 150], [300, 75, 400, 150]] 
image 1: [[25, 45, 55, 160], [320, 55, 420, 180]] 
image 2: [[15, 45, 55, 160], [350, 45, 480, 200]] 

Reasoning and reflection: I can see that I am getting closer but the last forward move is blocked.
The candidate computer is at [15, 45, 55, 160]. \
There could be another computer at [350, 45, 480, 200], but it is more far away from me so I will visit [15, 45, 55, 160] first. \
That means there is an obstacle in front of me and I can see it is probably a small chair. \
Following the strategy, I need to address the invalid action now since it is an obstacle blocking from moving closer in the direction of target object. The computer is at my left side in the front, \
to bypass this obstacle, I plan to first move leftward again to create more space, then try moving forward once more to get closer to the desk and the computer.
Executable plans: [[3, Move leftward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 6 is executed successfully; Step 7 is invalid because the StandardHeight is blocking the movement.

Candidate Target Entities location: 
image 1: [[25, 45, 55, 160], [320, 55, 420, 180]]
image 2: [[15, 45, 55, 160], [350, 45, 480, 200]]
image 3: [[0, 55, 35, 170], [350, 55, 450, 260]] 

Reasoning and reflection: I can see that the computer is still in my left front direction. \
The candidate computer is at [0, 55, 35, 170]. \
There could be another computer at [350, 55, 450, 260], but it is more far away from me so I will visit [0, 55, 35, 170] first. \
The chair continues to block my path. \
However, the chair continues to block my path. From the environment feedback, I observed that while my leftward movement was successful, my forward movement was still blocked. This indicates that I didn’t move far enough left to fully bypass the obstacle. To address this, I decide to move leftward two more times to ensure a clear path, then try moving forward again.
Executable plans: [[3, Move leftward by 0.25], [3, Move leftward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 8 is executed successfully; Step 9 is executed successfully; Step 10 is executed successfully.

Candidate Target Entities location: 
image 1: [[15, 45, 55, 160], [350, 45, 480, 200]]
image 2: [[0, 55, 35, 170], [350, 55, 450, 260]]
image 3: [[220, 75, 300, 150]] 

Reasoning and reflection: Now, I can see that the computer has shifted to my right front direction. 
The candidate computer is at [220, 75, 300, 150]. \
There are no longer any visible obstacles blocking my way. I deduce that the computer is now to my right because my previous leftward movements adjusted my position significantly to clear the earlier obstacle. Since the path is clear, I plan to approach the computer by moving forward three times and then rightward once to align myself next to it.
Executable plans: [[0, Move forward by 0.25], [0, Move forward by 0.25], [0, Move forward by 0.25], [2, Move rightward by 0.25]]

Environment Feedback: Step 11 is executed successfully; Step 12 is executed successfully; Step 13 is executed successfully; Step 14 is executed successfully.


Task Success

'''
,
'''
Human Instruction: Navigate to the book and be as close as possible.

Candidate Target Entities location: 
image 0: []

Reasoning and reflection: At the moment, I cannot see the book in my current field of view. \
According to the strategy you provided, I will rotate my view ninety degrees once at a time. \
Since rotation changes my vision, I will not plan any other actions until the book is visible. \
Executable plans: [[4, Rotate to the right by 90 degrees]]

Environment Feedback: Step 1 is executed successfully. The book is still not visible in my current field of view.

Candidate Target Entities location: 
image 0: [] 
image 1: []

Reasoning and reflection: I still cannot see the book in my current field of view. Following your instruction, I will rotate ninety degrees only once. As the rotation may change my view, I will not plan any additional actions until I locate the book.
Executable plans: [[4, Rotate to the right by 90 degrees]]

Candidate Target Entities location: 
image 0: []
image 1: []
image 2: [[160, 300, 200, 350], [230, 165, 240, 170]] 

Environment Feedback: Step 2 is executed successfully. \
The current candidate book is at [160, 300, 200, 350],
The could be another candidate book at [230, 165, 240, 170], but it is more far away from me so I will visit [160, 300, 200, 350] first. \
The book is now visible directly in front of me, placed on a shelf.

Reasoning and reflection: I have located the target object. The book is directly in front of me on a shelf. Following the strategy you provided, I will stop rotating now and start planning my navigation to get close to the book. Since the book is in front of me, I will first move two steps to the right to adjust my position, and then move forward three steps to approach it.
Executable plans: [[2, Move rightward by 0.25], [2, Move rightward by 0.25], [0, Move forward by 0.25], [0, Move forward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 3 to Step 7 are all executed successfully.

Candidate Target Entities location: 
image 0: [] 
image 1: [[160, 300, 200, 350], [230, 165, 240, 170]] 
image 2: [[200, 300, 300, 350], [300, 105, 340, 150]] 

Reasoning and reflection: I can now see that the book is directly in front of me, but I have not yet moved close enough to it.
The current candidate book is at [200, 300, 300, 350]. \
There could be another candidate book at [300, 105, 340, 150], but it is more far away from me so I will visit [200, 300, 300, 350] first. \
According to the standard, I should move closer to the book. Therefore, I decide to move forward three more times to reach the desired position.
Executable plans: [[0, Move forward by 0.25], [0, Move forward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 8, Step 9, and Step 10 are all executed successfully.

Candidate Target Entities location: 
image 0: [[160, 300, 200, 350], [230, 165, 240, 170]]
image 1: [[200, 300, 300, 350], [300, 105, 340, 150]]
image 2: [[200, 350, 300, 450], [300, 105, 340, 150]] 

Reasoning and reflection: I can now see that the book is directly in front of me. I am already very close to it, but the task has not success. \
I believe I need to move on to the next candidate at [300, 105, 340, 150].
According to the standard, I should move closer to the book. Therefore, I decide to move forward three more times to reach the desired position.
Executable plans: [[0, Move forward by 0.25], [0, Move forward by 0.25], [0, Move forward by 0.25]]

Environment Feedback: Step 11, Step 12, and Step 13 are all executed successfully.

Task Success


'''
] 



target_examples = [
'''
   Environmental Feedback: The last action succeeded. 
    
   Human Instruction:  I need a device on my floor to illuminate my workspace. Please navigate to that object and stay near it.
   
   Target Description:  A device on my floor to illuminate my workspace is a floorlamp. My target is a floorlamp.
   
   Visual Description: I can see the floorlamp is at left side in front of me, next to the TVStand. Around me is a Cabinet at my left side.
   
   Target Entity in JSON: {"target_name": ["floorlamp"], "blocking_name":[], "target_attribute": ["left side"], "target_relation": [["floorlamp", "next to", "TVStand"]], "related_objects": ["Cabinet", "TVStand"]}
''',

''' 
    Environmental Feedback: The last action is invalid. The feedback is: Last action MoveRight is invalid. StandardCounterHeightWidth is blocking Agent 0 from moving by (-0.2500, 0.0000, 0.0000).
    
    Human Instruction: Find the rectangular yellowish object with a soft and smooth surface.

    Target Description: My target is an object.

    Visual Description: At the first image, I can see that the rectangular yellowish object is  with a soft and smooth surface located in my left front direction, placed on a desk. In front of me, slightly to the right, there is a chair blocking my direct path. 
                        At the second image, I can see that the rectangular yellowish object is  with a soft and smooth surface is located in my left front direction, placed on a desk. 

    Target Entity in JSON: {"target_name": ["object"],  "blocking_name": ["counter"], "target_attribute": ["rectangular", "yellowish"], "target_relation": [["object", "placed on", "desk"]], "related_objects": ["chair", "desk"]}
''', 

''' Environmental Feedback: The last action is invalid. The feedback is: Last action MoveAhead is invalid. OVENDOOR.002 is blocking Agent 0 from moving by (0.0000, 0.0000, -0.2500). 

    Human Instruction: Navigate to the book and be as close as possible.

    Target Description: My target is a book.
    
    Visual Description: My target is a book.
                        At the first image, I cannot see the book. 
                        At the second image, I can see the book is directly in front of me, placed on a shelf.

    Target Entity in JSON: {"target_name": ["book"], "blocking_name": ["oven door"], "target_attribute": [], "target_relation": [["book", "placed on", "shelf"]], related_objects": ["shelf"]}
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
                                          detection_box=self.config.get('detection_box', False),
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
                                            gd_checkpoint_path=self.gd_checkpoint_path, gd_config_path=self.gd_config_path)

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
        'n_shots': 10,
        'down_sample_ratio': 1.0,
        'model_type': 'remote', # 'local', 
        'language_only': 0,
        'exp_name': 'history_laser_bbox_0.1_with_bbox_with_text',
        'chat_history': 0, 
        'detection_box': True,
        'multiview': False, 
        'eval_sets': ['base', 'spatial', 'visual_appearance', 'common_sense', 'complex_instruction', 'long_horizon'], 
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
        'gd_config_path': '/home/asethi04/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        'gd_checkpoint_path': '/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth'
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    evaluator = EB_AlfredEvaluator(config)
    evaluator.evaluate_main()




