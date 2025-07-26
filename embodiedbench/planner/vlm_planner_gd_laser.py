import torch
import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_config.generation_guide import (
    llm_generation_guide,
    vlm_target_generation_guide,
)
from embodiedbench.planner.planner_utils import (
    local_image_to_data_url,
    laser_template,
    laser_lang_template,
    fix_json,
)
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger
from PIL import Image

import copy
import matplotlib.pyplot as plt

# SAM2 imports
import sys
sam2_path = "/home/asethi04/sam2"
if sam2_path not in sys.path:
    sys.path.append(sam2_path)
    
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError as e:
    print(f"SAM2 not available: {e}")
    SAM2_AVAILABLE = False


template = laser_template

template_lang = laser_lang_template

MESSAGE_WINDOW_LEN = 5

import sys

# Use the pip-installed groundingdino-py package
try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    GROUNDINGDINO_AVAILABLE = True
    print("Using pip-installed groundingdino-py")
except ImportError as e:
    print(f"groundingdino-py not available: {e}")
    GROUNDINGDINO_AVAILABLE = False

import torch
import math

import numpy as np


model_lib_path = "/home/asethi04/LASER/LASER-unified/src/models"
sys.path.append(model_lib_path)

from llava_clip_model_v3 import PredicateModel as PredicateModel_v3

from collections import defaultdict

import json, textwrap, re, yaml          # pip install pyyaml
def robust_load_json_full(raw: str):
    """
    Heuristically clean up a model reply and return a Python object.
    Works in this order:
      1. direct ``json.loads``            (fast-path)
      2. json-after-fixups               (escape badly placed quotes)
      3. yaml.safe_load (very forgiving)
    Raises the last JSON error if every attempt fails.
    """

    # --- 0) quick path -------------------------------------------------
    try:
        return json.loads(raw)
    except Exception as strict_err:
        pass                      # fallthrough to the repair pipeline

    txt = raw.translate(str.maketrans({
        "“": '"', "”": '"',       # fancy → ASCII quotes
        "‘": "'", "’": "'"
    }))

    # -- 1) stray quotes OUTSIDE strings → replace by apostrophe --------
    txt = re.sub(r'(?<![\\"])("(?![\s]*[,:}\]]))', "'", txt)

    # -- 2) ':'  'value'   →   ":" "value"
    txt = re.sub(r'(":[ \t]*)\'', r'\1"', txt)

    # -- 3) escape single quotes inside strings ------------------------
    def _esc_single(m):
        return re.sub(r"(?<!\\)'", r"\\'", m.group(0))
    txt = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', _esc_single, txt)

    # -- 4) **escape double quotes that sit INSIDE a double-quoted value**
    def _esc_double(m):
        inner = m.group(1)
        inner_fixed = re.sub(r'(?<!\\)"', r'\\"', inner)
        return f'"{inner_fixed}"'
    txt = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', _esc_double, txt)

    # -- 5) remove control characters ----------------------------------
    txt = re.sub(r"[\x00-\x1F]", "", txt)

    # ---------- try JSON again ----------------------------------------
    try:
        return json.loads(txt)
    except Exception as json_err:
        last_err = json_err

    # ---------- final fallback: YAML ----------------------------------
    try:
        return yaml.safe_load(txt)
    except Exception:
        # give the *JSON* error – easier to understand for callers
        raise last_err

def robust_load_json(raw: str):
    """
    Best-effort loader for the sometimes messy JSON coming from the VLM.

    Strategy
    --------
    1.  plain ``json.loads``                                     (fast path)
    2.  sanitise obvious issues (un-escaped quotes, raw \n \r \t)
    3.  strip ``` fences   ➜  sanitise again
    4.  final fall-back:   *extract* the plan list with RegEx and
        build a minimal JSON around it so that downstream code can continue
        even if the rest of the payload is hopelessly malformed.

    Will raise the last JSONDecodeError only if **all** four passes fail.
    """
    # ---- 1) fast path --------------------------------------------------------
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e1:
        pass

    # ---- 2) sanitise & retry -------------------------------------------------
    try:
        return json.loads(_sanitize(raw))
    except json.JSONDecodeError as e2:
        pass

    # ---- 3) remove markdown fences and retry --------------------------------
    try:
        stripped = re.sub(r"```(?:json)?|```", "", raw, flags=re.I).strip()
        return json.loads(_sanitize(stripped))
    except json.JSONDecodeError as e3:
        pass

    # ---- 4) last-ditch: regex-pull a plan list ------------------------------
    plan_pat = re.compile(
        r'"(?:executable_plan|plan)"\s*:\s*(\[[^\]]*\])', re.S | re.I
    )
    m = plan_pat.search(raw)
    if not m:
        # propagate the *original* error – easiest for debugging
        raise e1

    plan_txt = m.group(1)
    # minimal valid JSON object
    return json.loads(f'{{"plan": {plan_txt}}}')


# ──────────────────────────────────────────────────────────────────────────────
#  _sanitize  – escape the two big culprits:    
#     • stray " inside strings      →   \"
#     • bare control chars          →   \n / \r / \t
# ──────────────────────────────────────────────────────────────────────────────
def _sanitize(text: str) -> str:
    buf, in_str, i, L = [], False, 0, len(text)
    while i < L:
        ch = text[i]

        # toggle / escape stray quotes ----------------------------------------
        if ch == '"':
            esc = i and text[i - 1] == '\\'
            if not esc:
                in_str = not in_str
            else:
                buf.append(ch)
                i += 1
                continue

        # if we *are* inside a JSON string …
        if in_str:
            # 1) un-escaped quotes that appear *within* the string -------------
            if ch == '"' and (i + 1 < L and text[i + 1] not in ',:}]\r\n\t '):
                buf.append('\\')

            # 2) raw control characters  --------------------------------------
            if ch in '\n\r\t':
                buf.append('\\' + {'\n': 'n', '\r': 'r', '\t': 't'}[ch])
                i += 1
                continue

        buf.append(ch)
        i += 1

    return ''.join(buf)
    
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

def format_state_descriptions(sg_info, current_obj, current_attr, current_rel):
    # Extract basic elements
    target_desc = sg_info.get("target_state", "No target state description provided.")
    current_desc = sg_info.get("current_state", "No current state description provided.")
    target_objects = sg_info.get("target_objects", [])
    target_attributes = sg_info.get("target_attributes", [])
    target_relations = sg_info.get("target_relations", [])

    target_sg_str = format_scene_graph(target_objects, target_attributes, target_relations)
    
    current_rel = [rel for _, _, rel in current_rel]
    current_sg_str = format_scene_graph(current_obj, current_attr, current_rel)
    
    sg_str = ""
    # if not current_sg_str == "":
    #     sg_str += f"The current scene graph: {current_sg_str}\n"
    # if not target_sg_str == "":
    #     sg_str += f"The target state: {target_sg_str}\n"
        
    sg_str += (f"\n\n The target state: {target_desc}\n" f"The current state: {current_desc}\n")
    # sg_str += f"The explanation is: {sg_info['explanation']} \n\n"
    
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

def load_model(model_dir, model_name, epoch, device):
    model_name = model_name + f".{epoch}.model"
    predicate_model = torch.load(
        os.path.join(model_dir, model_name), map_location=device, weights_only=False
    )
    return predicate_model


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
    mask[ymin : ymax + 1, xmin : xmax + 1, :] = True

    return mask

def convert_gd_boxes_to_pixel(boxes, img_width, img_height):
    """
    Convert GroundingDINO boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
    """
    if boxes is None or len(boxes) == 0:
        return []
    
    converted = []
    for box in boxes:
        if hasattr(box, 'cpu'):
            box = box.cpu().numpy()
        cx, cy, w, h = box
        # Convert from center format to corner format and denormalize
        x1 = int((cx - w/2) * img_width)
        y1 = int((cy - h/2) * img_height)
        x2 = int((cx + w/2) * img_width)
        y2 = int((cy + h/2) * img_height)
        converted.append([x1, y1, x2, y2])
    
    return converted

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    return result if result.endswith(".") else result + "."


def get_valid_bbox_val(v, b):
    return min(max(0, math.floor(v)), b)


def post_process_detections(det, classes):
    detections = det[0]
    class_ids = [
        0 if class_id is None else class_id for class_id in detections.class_id
    ]
    det.class_names = [classes[class_id] for class_id in class_ids]
    detections.xyxy = np.concatenate((detections.xyxy, det.xyxy), axis=0)
    detections.class_names = detections.class_names + det.class_names
    return detections


def list_depth(lst):
    """Calculates the depth of a nested list."""
    if not (isinstance(lst, list) or isinstance(lst, torch.Tensor)):
        return 0
    elif (isinstance(lst, torch.Tensor) and lst.shape == torch.Size([])) or (
        isinstance(lst, list) and len(lst) == 0
    ):
        return 1
    else:
        return 1 + max(list_depth(item) for item in lst)


def normalize_prompt(points, labels):

    if list_depth(points) == 3:
        points = torch.stack([p.unsqueeze(0) for p in points])
        labels = torch.stack([l.unsqueeze(0) for l in labels])
    return points, labels


# modify this function to take in a bbox label and show it


def show_box(box, ax, object_id, label="hi"):
    if len(box) == 0:
        return

    cmap = plt.get_cmap("gist_rainbow")
    cmap_idx = 0 if object_id is None else object_id
    color = list(cmap((cmap_idx * 47) % 256))
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=0.5)
    )

    if label is not None:
        ax.text(x0, y0 - 5, label, color="black", fontsize=10, alpha=color[3])


def save_prompts_one_image(frame_image, boxes, points, labels, save_path):
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Display the frame image
    ax.imshow(frame_image)
    ax.axis("off")
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

    # for object_id, (point_ls, label_ls) in enumerate(zip(points, labels)):
    #     if not len(point_ls) == 0:
    #         show_points(point_ls.cpu(), label_ls.cpu(), ax, object_id=object_id)
    # Show the plot
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def check_device_status(self):
    """Check and print device status for all models"""
    print("\n" + "="*50)
    print("DEVICE STATUS CHECK")
    print("="*50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("-" * 30)
    
    # Check GroundingDINO model device
    if hasattr(self, 'grounding_model') and self.grounding_model is not None:
        try:
            # Get the device of the first parameter
            device = next(self.grounding_model.parameters()).device
            print(f"GroundingDINO model device: {device}")
        except Exception as e:
            print(f"GroundingDINO model device: Unable to determine ({e})")
    else:
        print("GroundingDINO model: Not available")
    
    # Check SAM2 model device
    if hasattr(self, 'sam2_predictor') and self.sam2_predictor is not None:
        try:
            device = next(self.sam2_predictor.model.parameters()).device
            print(f"SAM2 model device: {device}")
        except Exception as e:
            print(f"SAM2 model device: Unable to determine ({e})")
    else:
        print("SAM2 model: Not available")
    
    # Check LASER predicate model device
    if hasattr(self, 'predicate_model') and self.predicate_model is not None:
        try:
            device = next(self.predicate_model.parameters()).device
            print(f"LASER predicate model device: {device}")
        except Exception as e:
            print(f"LASER predicate model device: Unable to determine ({e})")
    else:
        print("LASER predicate model: Not available")
    
    # Check main VLM model (if it has device info)
    if hasattr(self, 'model') and self.model is not None:
        if hasattr(self.model, 'device'):
            print(f"Main VLM model device: {self.model.device}")
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
            print(f"Main VLM model device: {self.model.model.device}")
        else:
            print("Main VLM model device: Unable to determine (likely remote/API)")
    else:
        print("Main VLM model: Not available")
    
    print("="*50)

class VLMPlanner:
    def __init__(
        self,
        model_name,
        model_type,
        actions,
        system_prompt,
        examples,
        n_shot=0,
        obs_key="head_rgb",
        chat_history=False,
        language_only=False,
        use_feedback=True,
        multistep=0,
        tp=1,
        kwargs={},
        scene_graph_examples="",
        gd_config_path="",
        gd_checkpoint_path="",
        boundingbox = False, 
        blockingobj = False,
        scene_graph_text = False,
        gd_only = False,
        top_k=1,
        aggr_thres=0.3,
        gd_box_threshold = 0.1,
        gd_text_threshold = 0.1,
        rel_thres=0.3,
        # Add SAM2 parameters
        use_sam2=False,
        sam2_config_path="",
        sam2_checkpoint_path="",
    ):
        
        self.last_visual_description = ""
        self.last_reasoning = ""
        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.scene_graph_examples = scene_graph_examples[:n_shot]
        self.n_shot = n_shot
        self.chat_history = (
            chat_history  # whether to includ all the chat history for prompting
        )
        self.set_actions(actions)
        self.model_type = model_type
        if model_type == "custom":
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp, task_type="alfred")

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.processed_images = {}
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop("action_key", "action_id")
        self.gd_box_threshold = gd_box_threshold
        self.gd_text_threshold = gd_text_threshold
        self.rel_thres = rel_thres
        
        self.scene_graph_prompt = f'''## You are a robot operating in a home. You are given an image and observe the environment for performing the given task. \
            The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'}\
            The input text given to you contains the previous action history, and please deduce the current state from both visual image and previous action histories. \
            Please deduce the object names with a synonym if the descriptions is verbose or functional, or you cannot find it in the scene. 
            For example, substitute "freshly baked baguette" to "loaves", "portable device to access the internet" to "laptop", "light-emiting device on my desk to provide light for my work area" to "desk lamp".
            You are supposed to output in JSON.'''
            
        self.scene_graph_prompt_prefix = "\n A missing scene graph means the object is not clearly identifiable from the scene. It does not necessarily means this object is not in the scene. \n"
            

        # Initialize GroundingDINO with groundingdino-py package
        if GROUNDINGDINO_AVAILABLE:
            try:
                # For groundingdino-py, we need to download and load the model
                # The model will be downloaded automatically if not present
                from groundingdino.util.inference import load_model as load_grounding_model
                import groundingdino
                import os
                
                # Use the correct config path from the installed package
                package_dir = os.path.dirname(groundingdino.__file__)
                model_config_path = os.path.join(package_dir, "config", "GroundingDINO_SwinT_OGC.py")
                
                # Check if weights file exists, if not download it
                model_checkpoint_path = "/home/asethi04/GroundingDINO/checkpoints/groundingdino_swint_ogc.pth"
                if not os.path.exists(model_checkpoint_path):
                    print("Model weights not found. Please download them using:")
                    print("wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
                    self.grounding_model = None
                    self.gd_available = False
                else:
                    # Try to load with groundingdino-py's load_model function
                    self.grounding_model = load_grounding_model(model_config_path, model_checkpoint_path)
                    print("GroundingDINO initialized successfully with groundingdino-py package")
                    self.gd_available = True
            except Exception as e:
                print(f"Failed to initialize GroundingDINO: {e}")
                print("This is expected if model weights are not downloaded. GroundingDINO will be disabled.")
                print("To download weights, run: wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
                self.grounding_model = None
                self.gd_available = False
        else:
            self.grounding_model = None
            self.gd_available = False
            print("GroundingDINO not available - some functionality will be limited")
        
        # Add SAM2 initialization
        self.use_sam2 = use_sam2 and SAM2_AVAILABLE
        if self.use_sam2:
            try:
                # Don't mess with Hydra initialization - SAM2 already handles this in its __init__.py
                # Just build the SAM2 model directly using the build_sam2 function
                self.sam2_predictor = SAM2ImagePredictor(
                    build_sam2(sam2_config_path, sam2_checkpoint_path, device="cuda:0", apply_postprocessing=False)
                )
                print("SAM2 initialized successfully")
            except Exception as e:
                print(f"SAM2 initialization failed: {e}")
                self.use_sam2 = False
                self.sam2_predictor = None
        else:
            self.sam2_predictor = None
            if use_sam2 and not SAM2_AVAILABLE:
                print("SAM2 requested but not available")
        
        self.boundingbox = boundingbox
        self.blockingobj = blockingobj
        self.scene_graph_text = scene_graph_text
        self.gd_only = gd_only
        self.aggr_thres = aggr_thres
        self.k = top_k
        
        all_model_dir = "/home/asethi04/common-data/jiani_common/llava_models"
        model_dir = os.path.join(all_model_dir, f"ensemble-02-10")
        model_name = "ensemble-2025-02-10-14-57-22"
        epoch = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # try:
        #     self.predicate_model = load_model(model_dir, model_name, epoch, device)
        # except Exception as e:
        #     print(f"Warning: Failed to load predicate model: {e}")
        #     print("Continuing without predicate model - some functionality may be limited")
        #     self.predicate_model = None
        clip_model_name = "openai/clip-vit-base-patch16"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predicate_model = PredicateModel_v3(hidden_dim = 128, num_top_pairs=1, device=device, model_name=clip_model_name).to(device)
        
        self.common_objects = ["wall", "door", "cabinet", "stove", "counter", "window", "other"]

    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ""
        for i in range(len(available_actions)):
            available_action_str += (
                "\naction id " + str(i) + ": " + str(available_actions[i])
            )
            if i < len(available_actions) - 1:
                available_action_str += ", "
        return available_action_str

    def process_prompt(self, user_instruction, prev_act_feedback=[]):
        last_env_feedback = (
            prev_act_feedback[-1][1] if len(prev_act_feedback) else "N/A"
        )              
        user_instruction = user_instruction.rstrip(".")
        failure = False
        blocking_feedback = ""
        
        if not len(prev_act_feedback) == 0:
            last_feedback = prev_act_feedback[-1]
            if last_feedback[0] != 133:
                failure = True
                blocking_feedback = last_feedback[1]
                
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                
                prompt = self.system_prompt.format(
                    len(self.actions) - 1,
                    self.available_action_str,
                    "\n\n".join(
                        [
                            f"## Task Execution Example {i}: \n {x}"
                            for i, x in enumerate(self.examples[: self.n_shot])
                        ]
                    ),
                )
                scene_graph_prompt = self.scene_graph_prompt + "\n\n".join(
                    [
                        f"## Vision Recognition Example {i}: \n {x}"
                        for i, x in enumerate(self.scene_graph_examples)
                    ]
                )
            else:

                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(
                    len(self.actions) - 1, self.available_action_str, ""
                )

            prompt += f"\n\n## Now the human instruction is: {user_instruction}."

            scene_graph_prompt += (
                f"\n\n## Now the human instruction is: {user_instruction}."
            )

            if self.language_only:
                prompt += f" You are supposed to output in json. Note that nested quotes is not allowed. Please use single quote within double quotes properly. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            else:
                prompt += f" You are supposed to output in json. Note that nested quotes is not allowed. Please use single quote within double quotes properly. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."

        elif self.chat_history:
            
            # This is to support the sliding window feature
            if self.n_shot >= 1:
                scene_graph_prompt = self.scene_graph_prompt + '\n\n'.join([f'## Vision Recognition Example {i}: \n {x}' for i,x in enumerate(self.scene_graph_examples)])
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f"The human instruction is: {user_instruction}."
            prompt += "\n\n The action history:"
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += "\nStep {}, action id {}, {}, env feedback: {}".format(
                        i,
                        action_feedback[0],
                        self.actions[action_feedback[0]],
                        action_feedback[1],
                    )
                else:
                    prompt += "\nStep {}, action id {}, {}".format(
                        i, action_feedback[0], self.actions[action_feedback[0]]
                    )
            if failure:
                scene_graph_prompt += f"The last action is invalid, and the feedback is: {blocking_feedback} "
            else:
                scene_graph_prompt += f"The last action succeeds. "
                
            scene_graph_prompt += f'In the explanation field, please include what are the previous actions that leads you to this description. Also provide candidate related objects if one thing is not found. \n\n## Now the human instruction is: {user_instruction}.'

            if self.language_only:
                prompt += f"""\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions."""
            else:
                prompt += f"""\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions."""
        else:
            if self.n_shot >= 1:
                scene_graph_prompt = self.scene_graph_prompt + "\n\n".join(
                    [
                        f"## Vision Recognition Example {i}: \n {x}"
                        for i, x in enumerate(self.scene_graph_examples)
                    ]
                )
                prompt = self.system_prompt.format(
                    len(self.actions) - 1,
                    self.available_action_str,
                    "\n\n".join(
                        [
                            f"## Task Execution Example  {i}: \n {x}"
                            for i, x in enumerate(self.examples[: self.n_shot])
                        ]
                    ),
                )
            else:
                scene_graph_prompt = self.scene_graph_prompt
                prompt = self.system_prompt.format(
                    len(self.actions) - 1, self.available_action_str, ""
                )
            prompt += f"\n\n## Now the human instruction is: {user_instruction}."
            
            action_history_prompt = "\n\n The action history:"
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    action_history_prompt += "\nStep {}, action id {}, {}, env feedback: {}".format(
                        i,
                        action_feedback[0],
                        self.actions[action_feedback[0]],
                        action_feedback[1],
                    )
                else:
                    action_history_prompt += "\nStep {}, action id {}, {}".format(
                        i, action_feedback[0], self.actions[action_feedback[0]]
                    )
            
            prompt += action_history_prompt
            scene_graph_prompt += action_history_prompt
                    
            if failure:
                scene_graph_prompt += f"The last action is invalid, and the feedback is: {blocking_feedback} "
            else:
                scene_graph_prompt += f"The last action succeeds. "
                
            # scene_graph_prompt += (
            #     # f"\n\nEnvironmental Feedback: {last_env_feedback}"
            #     f"\n\nLast Visual Description: {self.last_visual_description}\n"
            # )
            
            # scene_graph_prompt += (
            #     # f"\n\nEnvironmental Feedback: {last_env_feedback}"
            #     f"\n\nReasoning and Reflection: {self.last_reasoning}\n"
            # )
            
            scene_graph_prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            if self.language_only:
                prompt += f"""\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions."""
            else:
                prompt += f"""\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions."""
        
        return prompt, scene_graph_prompt

    def get_message(self, image, prompt, messages=[]):
        if self.language_only:
            return messages + [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
        else:
            if type(image) == str:
                image_path = image
            elif type(image) == list:
                image_path = image[-1]
            else:
                image_path = "./evaluation/tmp_{}.png".format(len(messages) // 2)
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

            if self.multistep:  # handle multiple images
                ind = int(image_path.split("step_")[-1].strip(".png"))
                content = [{"type": "text", "text": prompt}]
                for i in range(max(ind - self.multistep + 1, 0), ind + 1):
                    temp_path = (
                        "".join(image_path.split("step_")[:-1]) + f"step_{str(i)}.png"
                    )
                    temp_data_url = local_image_to_data_url(image_path=temp_path)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": temp_data_url,
                            },
                        }
                    )
            else:
                data_url = local_image_to_data_url(image_path=image_path)
                content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                    {"type": "text", "text": prompt},
                ]

            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def reset(self):
        # at the beginning of the episode
        self.last_visual_description = ""
        self.last_reasoning = ""
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r"\*\*\d+\*\*"
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip("*"))
        else:
            print("random action")
            action = np.random.randint(len(self.actions))
        return action

    def json_to_action(self, output_text, json_key="executable_plan"):
        valid = True
        try:
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            if not len(action):
                print("empty plan, stop here")
                action = -2
            else:
                # keep action valid
                for i, act in enumerate(action):
                    if act >= len(self.actions) or act < 0:
                        print("found invlid action")
                        if i == 0:
                            action = -1
                        else:
                            action = action[:i]
                        break
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
            valid = False
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            self.output_json_error += 1
            action = -1
            valid = False
        return action, valid
    
    def json_to_action_alt(
            self,
            output_text: str,
            json_keys: tuple = ("executable_plan", "plan"),
        ):
        """
        Parse the model’s JSON and convert it to a list of action-ids.

        • Accept either ``"executable_plan"`` or ``"plan"``.  
        • Trust an ``action_id`` only if it is in range *and* its name matches
        the supplied ``action_name`` after normalisation.  
        • Otherwise attempt to resolve the step purely by its ``action_name``;
        **the name must exactly (case-insensitive) match one entry in
        ``self.actions``.**  
        • Returns  (action_list | int_flag ,  valid_bool)
            -2  … empty plan / stop  
            -1  … could not resolve a step           

        Normalisation = lower-case & collapse multiple spaces.
        """
        def norm(txt: str) -> str:                # helper
            return " ".join(txt.lower().split())

        # build fast lookup: normalised name  ->  id
        name2id = {norm(a): i for i, a in enumerate(self.actions)}

        try:
            # bad = output_text
            # # quick sanitiser – remove any quote directly before a t in isn"t ➜ isn't
            # bad = re.sub(r'isn"t', "isn't", bad)

            # obj = json.loads(bad) 
            def parse_json_with_fixes(json_str):
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try fixing common escape issues
                    fixed_str = json_str.replace("\\'", "'")
                    return json.loads(fixed_str)
  
            obj = parse_json_with_fixes(output_text)

            # locate a list plan (first key that exists & is a list)
            plan = next((obj[k] for k in json_keys if isinstance(obj.get(k), list)), None)
            if plan is None:
                raise ValueError("no plan key found")

            action_ids = []
            valid = True
            for step in plan:
                if not isinstance(step, dict):
                    raise ValueError("plan step not dict")

                aid = step.get(self.action_key)          # e.g. "action_id"
                anm = step.get("action_name")
                nm  = norm(anm) if isinstance(anm, str) else None

                resolved = None

                # ── (1)  validate a provided id ─────────────────────────────────
                if (isinstance(aid, int)
                        and 0 <= aid < len(self.actions)
                        and nm == norm(self.actions[aid])):
                    resolved = aid

                # ── (2)  otherwise resolve by exact name match ──────────────────
                if resolved is None and nm in name2id:
                    resolved = name2id[nm]

                if resolved is None:      # could not resolve → invalid plan
                    # valid = False
                    break

                action_ids.append(resolved)

            if not action_ids:            # empty => stop signal
                action = -2
                valid  = True
            elif valid is False:          # at least one step unresolved
                action = -1
            else:
                action = action_ids
                valid  = True

        except Exception as e:
            print("Could not parse model JSON:", e)
            self.output_json_error += 1
            action = -1
            valid  = False

        return action, valid

    def predict_gd(self, image, classes, box_threshold=0.5, text_threshold=0.4):
        if not self.gd_available or self.grounding_model is None:
            print("GroundingDINO not available, returning empty results")
            return [], []
            
        # Convert classes list to caption
        caption = ". ".join(classes)
        processed_caption = preprocess_caption(caption)
        
        try:
            # Use groundingdino-py API
            # Convert image to the format expected by groundingdino-py
            from PIL import Image as PILImage
            import torchvision.transforms as T
            
           # After your RGB conversion
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Convert to PIL Image if it's a numpy array
            if isinstance(rgb_image, np.ndarray):
                pil_image = PILImage.fromarray(rgb_image)
            else:
                pil_image = rgb_image  # Already a PIL Image

            # Create the transform
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization

            ])

            # Convert to tensor
            image_tensor = transform(pil_image)
                
           
            
            # Use groundingdino-py's predict function
            boxes, logits, phrases = predict(
                model=self.grounding_model,
                image=image_tensor,
                caption=processed_caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Convert results to the expected format
            labels = phrases if phrases is not None else []
            
            # Convert boxes to tensor if they aren't already
            if boxes is not None and len(boxes) > 0:
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes)
            else:
                boxes = torch.tensor([])        
        except Exception as e:
            print(f"Error in GroundingDINO prediction: {e}")
            return [], []
                
        return labels, boxes
    
    
    
    # def predict_laser(self, 
    #               image, 
    #               boxes,  # Now expects pixel coordinates as list of [x1, y1, x2, y2]
    #               masks=None,  # Optional masks parameter
    #               target_names=None,
    #               target_relation_kws=None,
    #               target_relation_preds=None,
    #               query_names=None,
    #               query_relations=None,
    #               query_relation_kws=None,
    #               query_attributes=None,
    #               ):
        
    #     # Preprocess query attributes for the case where one object having multiple attribute
    #     query_attribute_ls = [[attr, f"not {attr}"] for attr in query_attributes]
    #     prob_labels, selected_object_bboxes, identified_rels = [], [], []
        
    #     if not len(boxes) == 0 and self.predicate_model is not None:
    #         height, width = image.shape[0], image.shape[1]
            
    #         # Use provided masks or generate from bboxes
    #         if masks is not None and len(masks) > 0:
    #             # Ensure masks have the right shape
    #             batched_pred_masks = []
    #             for mask in masks:
    #                 if isinstance(mask, torch.Tensor):
    #                     mask = mask.cpu().numpy()
    #                 # Add channel dimension if needed
    #                 if mask.ndim == 2:
    #                     mask = mask[:, :, np.newaxis]
    #                 batched_pred_masks.append(mask)
    #         else:
    #             # Fall back to generating masks from bboxes
    #             batched_pred_masks = [bbox_to_mask(bbox, height, width) for bbox in boxes]
            
    #         batched_object_ids = [[0, 0, i] for i in range(len(batched_pred_masks))]
            
    #         batched_image_cate_probs, _, _, _ = \
    #             self.predicate_model(
    #                 batched_video_ids=[0],
    #                 batched_videos=[image],
    #                 batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
    #                 batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
    #                 batched_names=[query_names],  # Dataset-wise categorical labels
    #                 batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
    #                 batched_unary_kws=[[]],  # Dataset-wise unary predicate labels
    #                 batched_binary_kws=[[]],  # Dataset-wise binary predicate labels
    #                 batched_obj_pairs=[],  # Ground truth binary relations
    #                 batched_video_splits=[0],  # [number of videos]
    #                 batched_binary_predicates=[[]],  # None indicates inference time
    #             )
                    
    #         all_unary_res = []
    #         for unary in query_attribute_ls:
    #             _, batched_image_unary_probs, _, _ = \
    #                 self.predicate_model(
    #                     batched_video_ids=[0],
    #                     batched_videos=[image],
    #                     batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
    #                     batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
    #                     batched_names=[query_names],  # Dataset-wise categorical labels
    #                     batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
    #                     batched_unary_kws=[unary],  # Dataset-wise unary predicate labels
    #                     batched_binary_kws=[[]],  # Dataset-wise binary predicate labels
    #                     batched_obj_pairs=[],  # Ground truth binary relations
    #                     batched_video_splits=[0],  # [number of videos]
    #                     batched_binary_predicates=[None],  # None indicates inference time
    #                 )
    #             all_unary_res.append(batched_image_unary_probs)
            
    #         # Combine the unary and object name information together 
    #         # Step 1: reformat the object name, attributes, and relations into better format
    #         predicted_name = {}
    #         for (oid, name) in batched_image_cate_probs[0]:
    #             prob = batched_image_cate_probs[0][(oid, name)].item()
    #             if not oid in predicted_name:
    #                 predicted_name[oid] = []
    #             predicted_name[oid].append((prob, name))
            
    #         for oid in predicted_name:
    #             predicted_name[oid] = sorted(predicted_name[oid], reverse=True)   
                    
    #         reformatted_unary_res = {}
    #         for unary_res in all_unary_res:
    #             for  (fid, oid, kw) in unary_res[0].keys():
    #                 if not oid in reformatted_unary_res:
    #                     reformatted_unary_res[oid] = {}
    #                 if kw in query_attributes:
    #                     reformatted_unary_res[oid][kw] = unary_res[0][(fid, oid, kw)].item()
            
    #         topk_obj_pair_info = get_topk_per_relation(predicted_name, binary_predicates=query_relations, top_k=30)
    #         top_obj_pairs = merge_object_pair_ids(topk_obj_pair_info)
            
    #         batched_image_rel_probs = {}
    #         if len(top_obj_pairs) > 0:
    #             _, _, batched_image_rel_probs, _ = \
    #                 self.predicate_model(
    #                     batched_video_ids=[0],
    #                     batched_videos=[image],
    #                     batched_masks=batched_pred_masks,  # batched_object_ids * video_height * video_width
    #                     batched_bboxes=boxes,  # batched_object_ids * dict<bboxes>
    #                     batched_names=[query_names],  # Dataset-wise categorical labels
    #                     batched_object_ids=batched_object_ids,  # [video_id, frame_id, object_id]
    #                     batched_unary_kws=[[]],  # Dataset-wise unary predicate labels
    #                     batched_binary_kws=[query_relation_kws],  # Dataset-wise binary predicate labels
    #                     batched_obj_pairs=top_obj_pairs,  # Ground truth binary relations
    #                     batched_video_splits=[0],  # [number of videos]
    #                     batched_binary_predicates=[query_relations],  # None indicates inference time
    #                 )
                
    #         # Step 3: Aggregate the object name and attribute predictions together
    #         aggr_res = {}            
            
    #         for oid in predicted_name:
    #             top_prob, name = predicted_name[oid][0]
    #             if not oid in aggr_res:
    #                 aggr_res[oid] = {}
                
    #             if name in target_names:
    #                 aggr_res[oid]['name'] = top_prob, name
    #             else:
    #                 # the top predicted name is not the target name
    #                 aggr_res[oid]['name'] = 0, "other"
                    
    #         for oid in reformatted_unary_res: 
                
    #             if not oid in aggr_res:
    #                 continue
                
    #             if not 'attr' in aggr_res[oid]:
    #                 aggr_res[oid]['attr'] = []
    #             for attr in reformatted_unary_res[oid]:
    #                 aggr_res[oid]['attr'].append((reformatted_unary_res[oid][attr], attr))
            
    #         if len(batched_image_rel_probs) > 0:
    #             rel_preds = batched_image_rel_probs[0]
    #         else:
    #             rel_preds = {}

    #         identified_rels = []
    #         for (_, (subj, obj), rel), p in rel_preds.items():
    #             subj_prob, pred_subj_name = predicted_name[subj][0]
    #             obj_prob, pred_obj_name = predicted_name[obj][0]
    #             rel_prob = p
    #             if (rel, pred_subj_name, pred_obj_name) in target_relation_preds:
    #                 prob = subj_prob * obj_prob * rel_prob
    #                 if prob > self.rel_thres:
    #                     identified_rels.append((prob, (rel, subj, obj), (rel, pred_subj_name, pred_obj_name)))
                
    #         # Thresholding
    #         avg_aggr_res = {}
    #         for oid in aggr_res:
    #             name_prob, name = aggr_res[oid]['name']
                
    #             # Target Name must rank at top 1
    #             if name_prob == 0:
    #                 continue
                
    #             # Attribute aggregation
    #             probs = [name_prob]

    #             if 'attr' in aggr_res[oid]:
    #                 probs += [p for p, _ in aggr_res[oid]['attr']]
                
    #             avg_val = sum(probs) / len(probs)
                    
    #             if avg_val > self.aggr_thres:
    #                 avg_aggr_res[oid] = avg_val
            
    #         # Relation aggregation
                
    #         # Only pick the top-k object according to the setup
    #         top_objects_probs = get_topk_per_object_name(aggr_res, avg_aggr_res, k=self.k)
            
    #         selected_objects = list(top_objects_probs.keys())
    #         # Handle pixel coordinate boxes (list of lists)
    #         selected_object_bboxes = [boxes[idx] for idx in selected_objects]
                                
    #         prob_labels = []
    #         for oid in top_objects_probs:
    #             prob, label = predicted_name[oid][0]
    #             prob_labels.append(f"{prob:.2f}::{label}")

    #     return prob_labels, selected_object_bboxes, identified_rels
    def predict_laser(self, 
                  image, 
                  boxes,  # Expects pixel coordinates as list of [x1, y1, x2, y2]
                  masks=None,  # Optional masks parameter
                  target_names=None,
                  query_names=None,
                  threshold=0.3,  # Probability threshold
                  ):
        prob_labels, selected_object_bboxes, selected_masks = [], [], []
        
        if not len(boxes) == 0 and self.predicate_model is not None:
            height, width = image.shape[0], image.shape[1]
            
            # Use provided masks or generate from bboxes
            if masks is not None and len(masks) > 0:
                batched_pred_masks = []
                for mask in masks:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    if mask.ndim == 2:
                        mask = mask[:, :, np.newaxis]
                    batched_pred_masks.append(mask)
            else:
                batched_pred_masks = [bbox_to_mask(bbox, height, width) for bbox in boxes]
            
            batched_object_ids = [[0, 0, i] for i in range(len(batched_pred_masks))]
            
            # Get categorical predictions only
            batched_image_cate_probs, _, _, _ = \
                self.predicate_model(
                    batched_video_ids=[0],
                    batched_videos=[image],
                    batched_masks=batched_pred_masks,
                    batched_bboxes=boxes,
                    batched_names=[query_names],
                    batched_object_ids=batched_object_ids,
                    batched_unary_kws=[[]],
                    batched_binary_kws=[[]],
                    batched_obj_pairs=[],
                    batched_video_splits=[0],
                    batched_binary_predicates=[[]],
                )
            
            # Process each object
            for oid in range(len(boxes)):
                # Find the best matching category for this object
                best_prob = 0
                best_name = None
                
                for (obj_id, name) in batched_image_cate_probs[0]:
                    if obj_id == oid:
                        prob = batched_image_cate_probs[0][(obj_id, name)].item()
                        if prob > best_prob:
                            best_prob = prob
                            best_name = name
                
                # Check if this object matches our criteria
                if best_name in target_names and best_prob > threshold:
                    prob_labels.append(f"{best_prob:.2f}::{best_name}")
                    selected_object_bboxes.append(boxes[oid])
                    # Also append the corresponding mask
                    selected_masks.append(batched_pred_masks[oid])
        
        return prob_labels, selected_object_bboxes, selected_masks
    
    def predict_sam2_masks(self, image, boxes):
        """
        Generate SAM2 masks from bounding boxes
        Args:
            image: RGB image array
            boxes: tensor of bounding boxes [N, 4] in format [x1, y1, x2, y2]
        Returns:
            masks: list of binary masks
        """
        if not self.use_sam2 or self.sam2_predictor is None:
            return []
            
        if len(boxes) == 0:
            return []
            
        try:
            # Set image for SAM2 predictor
            self.sam2_predictor.set_image(np.array(image))
            
            # Convert boxes to numpy if needed
            if isinstance(boxes, torch.Tensor):
                boxes_np = boxes.cpu().numpy()
            else:
                boxes_np = np.array(boxes)
                
            # Generate masks for all boxes
            masks, scores, _ = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=False,
            )
            
            # Squeeze masks if needed
            masks = [masks[i].squeeze(0) if masks[i].shape[0] == 1 else masks[i] for i in range(len(masks))]
            
            return masks
            
        except Exception as e:
            print(f"SAM2 prediction failed: {e}")
            return []
    
    def save_mask_contours_and_boxes(self, frame_image, masks, boxes, labels, save_path):
        """
        Save image with mask contours and bounding boxes
        """
        if len(masks) == 0 or len(boxes) == 0:
            return
            
        contour_img = frame_image.copy()
        
        # Draw mask contours
        for mask in masks:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Squeeze mask if needed
            if mask.ndim == 3:
                mask = mask.squeeze()
                
            segmentation = mask.astype(np.uint8)
            contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Use thin red line for better visibility
            cv2.drawContours(contour_img, contours, -1, color=(255, 0, 0), thickness=1)
        
        # Draw labels
        for i, (box, label) in enumerate(zip(boxes, labels)):
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            x0, y0, x1, y1 = map(int, box)
            
            # Draw label with subtle shadow for visibility
            # Shadow
            cv2.putText(contour_img, label, (x0+1, y0-4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            # Main text
            cv2.putText(contour_img, label, (x0, y0-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Save result
        cv2.imwrite(save_path, cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))
    
    def get_sg_kws(self, sg_info):
        

        
        target_objects = sg_info['target_objects']
        target_attributes = sg_info['target_attributes']   
        target_relations = sg_info['target_relations']
        current_objects = sg_info['current_objects']
        current_attributes = sg_info['current_attributes']
        current_relations = sg_info['current_relations']
        
        target_names = list(set([clean_name(object) for object in target_objects]))
        current_names = list(set([clean_name(object) for object in current_objects]))
        target_attribute_kws = list(set([ta[1] for ta in target_attributes if len(ta) == 2]))
        current_attribute_kws = list(set([ca[1] for ca in current_attributes if len(ca) == 2]))

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
               target_attribute_kws, current_attribute_kws, \
               target_relation_preds, current_relation_preds, \
               target_relation_kws, query_relation_kws, query_relation_preds

    def act_custom(self, prompt, obs):
        assert type(obs) == str  # input image path
        out = self.model.respond(prompt, obs)
        # fix common generated json errors
        out = out.replace("'", '"')
        out = out.replace('"s ', "'s ")
        out = out.replace('"re ', "'re ")
        out = out.replace('"ll ', "'ll ")
        out = out.replace('"t ', "'t ")
        out = out.replace('"d ', "'d ")
        out = out.replace('"m ', "'m ")
        out = out.replace('"ve ', "'ve ")
        out = out.replace("```json", "").replace("```", "")
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action = self.json_to_action_alt(out)
        self.planner_steps += 1
        return action, out

    def act(self, observation, user_instruction):

        if type(observation) == dict:
            obs = copy.deepcopy(observation[self.obs_key])
        else:
            # assert len(observation) == 1
            obs = copy.deepcopy(observation)  # input image path

        prompt, scene_graph_prompt = self.process_prompt(
            user_instruction, prev_act_feedback=self.episode_act_feedback
        )
        # some models do not support json scheme, add style into prompt
        if (
            "claude" in self.model_name
            or "InternVL" in self.model_name
            or "Qwen" in self.model_name
            or self.model_type == "custom"
        ):
            prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == "custom":
            return self.act_custom(prompt, obs)

        sg_message = self.get_message(obs, scene_graph_prompt)

        try:
            out = self.model.respond(sg_message, get_scene_graph=True)
        except Exception as e:
            print(e)
            if "qwen" in self.model_name:
                return (
                    -2,
                    """{
                    "visual_state_description": "qwen model generate empty action due to inappropriate content check",
                    "reasoning_and_reflection": "invalid json, random action",
                    "language_plan": "invalid json, random action"
                }""",
                )

        # Parse scene graph JSON with error handling
        try:
            sg_info = robust_load_json_full(out)
        except Exception as e:
            print(f"Scene graph JSON parsing failed: {e}")
            # Use default scene graph structure if parsing fails
            sg_info = {
                "target_state": "Unable to parse scene graph",
                "current_state": "Unable to parse scene graph", 
                "target_objects": [],
                "target_attributes": [],
                "target_relations": [],
                "current_objects": [],
                "current_attributes": [],
                "current_relations": [],
                "explanation": "JSON parsing failed, using default structure"
            }

        target_state = sg_info.get("target_state", "No target state available")
        current_state = sg_info.get("current_state", "No current state available")
        try:
            target_names, current_names, query_objects, \
            target_attribute_kws, current_attribute_kws, \
            target_relation_preds, current_relation_preds, \
                target_relation_kws, query_relation_kws, query_relation_preds = self.get_sg_kws(sg_info)
        except Exception as e:
            print(f"Error processing scene graph keywords: {e}")
            # Use empty defaults if scene graph processing fails
            target_names = []
            current_names = []
            query_objects = self.common_objects.copy()
            target_attribute_kws = []
            current_attribute_kws = []
            target_relation_preds = set()
            current_relation_preds = set()
            target_relation_kws = set()
            query_relation_kws = []
            query_relation_preds = []
        scene_graph_prompt = []
        sg_text = ""
         
        if type(obs) == str:
            obs = [obs]
        if type(observation) == str:
            observation = [observation]

        for image_id, image_path in enumerate(observation):
            image_path_ls = image_path.split("/")
            image_dir, image_name = image_path_ls[:-1], image_path_ls[-1]
            image_name = image_name.split(".")[0]
            bbox_file_name = image_name + '_bbox.png'

            bbox_path = os.path.join(*image_dir, bbox_file_name)

            prompt_image_path = image_path

            if image_path in self.processed_images:
                processed_image_path = self.processed_images[image_path]
                if not processed_image_path in obs:
                    obs = [processed_image_path]
                    continue

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width = image.shape[0], image.shape[1]
            
            if self.gd_only:
                
                # Predict bounding box with grounding dino
                labels, boxes = self.predict_gd(image, target_names, box_threshold=self.gd_box_threshold, text_threshold=self.gd_text_threshold)
            
                target_labels, target_bboxes = [], []
                for label, box in zip(labels, boxes):
                    if label in target_names:
                        target_labels.append(label)
                        target_bboxes.append(box)
                        
                if len(target_bboxes) > 0:
                    target_bboxes = torch.stack(target_bboxes)
            
            else:
                # og_laels, og_boxes = self.predict_gd(image, gd_query_objects, box_threshold=0.4, text_threshold=0.4)
                labels, boxes = self.predict_gd(image, query_objects,
                                box_threshold=self.gd_box_threshold, 
                                text_threshold=self.gd_text_threshold)

                if len(boxes) != 0:
                    # Get correct image dimensions
                    img_height, img_width = image.shape[0], image.shape[1]
                    
                    # Convert boxes to pixel coordinates for LASER
                    boxes_pixel = convert_gd_boxes_to_pixel(boxes, img_width, img_height)
                    
                    # Generate SAM2 masks if enabled (SAM2 expects pixel coordinates)
                    sam2_masks = []
                    if self.use_sam2 and len(boxes_pixel) > 0:
                        # Convert to tensor for SAM2
                        boxes_pixel_tensor = torch.tensor(boxes_pixel, dtype=torch.float32)
                        sam2_masks = self.predict_sam2_masks(image, boxes_pixel_tensor)
                    
                    # # Call predict_laser with pixel coordinates and masks
                    # target_labels, target_bboxes, identified_rels = self.predict_laser(
                    #     image=image, 
                    #     boxes=boxes_pixel,  # Pixel coordinates as list
                    #     masks=sam2_masks if sam2_masks else None,  # Pass masks if available
                    #     target_names=target_names,
                    #     target_relation_kws=target_relation_kws,
                    #     target_relation_preds=target_relation_preds,
                    #     query_names=list(set(query_objects)),
                    #     query_attributes=target_attribute_kws,
                    #     query_relation_kws=query_relation_kws,
                    #     query_relations=query_relation_preds,
                    # )
                    # Call predict_laser with pixel coordinates and masks
                    target_labels, target_bboxes, identified_masks = self.predict_laser(
                        image=image, 
                        boxes=boxes_pixel,  # Pixel coordinates as list
                        masks=sam2_masks if sam2_masks else None,  # Pass masks if available
                        target_names=target_names,
                        query_names=list(set(query_objects)),  # All possible object names to check
                        threshold=self.aggr_thres  # Use the existing threshold from self
                    )
                else:
                    target_labels = []
                    target_bboxes = []
                    identified_rels = []

                # 4. Update visualization part:
                # Save visualization with boxes and optionally masks
                if not len(target_bboxes) == 0:
                    # Convert list of lists to tensor for visualization functions
                    target_bboxes_tensor = torch.tensor(target_bboxes, dtype=torch.float32)
                    
                    if self.use_sam2 and len(sam2_masks) > 0:
                        # Save with both masks and boxes
                        mask_bbox_path = bbox_path.replace('_bbox.png', '_mask_bbox.png')
                        self.save_mask_contours_and_boxes(
                            frame_image=image, 
                            masks=identified_masks,
                            boxes=target_bboxes_tensor, 
                            labels=target_labels,
                            save_path=mask_bbox_path
                        )
                        prompt_image_path = mask_bbox_path
                    else:
                        # Save with just boxes
                        save_prompts_one_image(
                            frame_image=image, 
                            boxes=target_bboxes_tensor, 
                            labels=["" for _ in target_bboxes], 
                            points=[], 
                            save_path=bbox_path
                        )    
                        prompt_image_path = bbox_path
                
                else:
                    target_bboxes = []
                    target_labels = []
            
            self.processed_images[image_path] = prompt_image_path

            # Arrange the prompt into vlm
            if self.scene_graph_text:
                try:
                    sg_text = format_state_descriptions(sg_info, target_labels, [], [])
                except Exception as e:
                    print(f"Error formatting scene graph text: {e}")
                    sg_text = f"Current state: {current_state}\nTarget state: {target_state}"
        
        if self.multistep:
            if not image_path == prompt_image_path:
                obs.append(prompt_image_path) # update to bboxed image path
        
        if self.scene_graph_text and sg_text is not None:

            marker = "Now the human instruction is"
            try:
                insert_idx = prompt.index(marker)      # first char of the marker
                prompt = prompt[:insert_idx] + sg_text + prompt[insert_idx:]
            except ValueError:
                # marker not found – fallback: append at the end
                prompt += sg_text
            
        
        else:
            obs = [prompt_image_path]
            
        obs = sorted(list(set(obs)))
        
        if not self.multistep:
            assert len(obs) == 1
            obs = obs[0]

        if len(self.episode_messages) == 0:
            self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(
                    obs, prompt, self.episode_messages
                )
            else:
                self.episode_messages = self.get_message(obs, prompt)

        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        if "gemini-1.5-pro" in self.model_name or "gemini-2.0-flash" in self.model_name:
            try:
                out = self.model.respond(self.episode_messages)
                time.sleep(10)
            except Exception as e:
                print("An unexpected error occurred:", e)
                time.sleep(20)
                out = self.model.respond(self.episode_messages)
        else:
            try:
                out = self.model.respond(self.episode_messages)
            except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != "local":
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
        # out = out.replace("'", '"')
        out = out.replace('"s ', "'s ")
        out = out.replace('"re ', "'re ")
        out = out.replace('"ll ', "'ll ")
        out = out.replace('"t ', "'t ")
        out = out.replace('"d ', "'d ")
        out = out.replace('"m ', "'m ")
        out = out.replace('"ve ', "'ve ")
        out = out.replace("```json", "").replace("```", "")
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action, valid = self.json_to_action_alt(out)
        
        if valid:
            try:
                def parse_json_with_fixes(json_str):
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try fixing common escape issues
                        fixed_str = json_str.replace("\\'", "'")
                        return json.loads(fixed_str) 
                plan = parse_json_with_fixes(out)
                self.last_reasoning = plan.get(
                        "reasoning_and_reflection", ""
                )
                self.last_visual_description = plan.get(
                        "visual_state_description", ""
                )
                
                plan['scene_graph'] = sg_info
                out_str = json.dumps(plan)
            except Exception as e:
                print(f"Error parsing final JSON output: {e}")
                self.last_visual_description = ""
                self.last_reasoning = ""
                out_str = '''{"visual_state_description":"json parsing failed", "reasoning_and_reflection":"json parsing failed",
                       "language_plan":"json parsing failed"}'''
        else:
            self.last_visual_description = ""
            self.last_reasoning = ""
            out_str = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
                   
        self.planner_steps += 1
        return action, out_str

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([info["action_id"], info["env_feedback"]])