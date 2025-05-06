import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
import math

import torch
import math
from collections import defaultdict, Counter
# from example_pred import entities, colors, shapes
from example_pred import entities

def compute_entropy(probs):
    """Compute entropy of a list of probabilities."""
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

def assign_entities_with_entropy(entity_list, predicted_probs):
    """
    Assign entity names to object ids using dynamic entropy-based selection.
    
    Args:
        entity_list: List[str] - target entity names (may contain duplicates)
        predicted_probs: List[((int, str), torch.Tensor)] - ((obj_id, name), prob)

    Returns:
        Dict[int, str] - mapping from obj_id to assigned name
    """
    # Count how many times each name is needed
    name_budget = Counter(entity_list)
    obj_ids = sorted(set(obj_id for (obj_id, _), _ in predicted_probs.items()))
    
    # Group predictions by obj_id
    dist_per_obj = defaultdict(list)
    for (obj_id, name), prob in predicted_probs.items():
        dist_per_obj[obj_id].append((name, prob))

    # Normalize probabilities per object
    for obj_id in dist_per_obj:
        total = sum(prob for _, prob in dist_per_obj[obj_id])
        if total > 0:
            dist_per_obj[obj_id] = [(name, prob / total) for name, prob in dist_per_obj[obj_id]]

    assigned = {}
    while len(assigned) < len(obj_ids):
        # Compute entropy per unassigned obj_id
        entropy_list = []
        for obj_id in obj_ids:
            if obj_id in assigned:
                continue
            probs = [prob for name, prob in dist_per_obj[obj_id] if name_budget[name] > 0]
            entropy = compute_entropy(probs) if probs else float('inf')
            entropy_list.append((entropy, obj_id))

        # Pick object with minimum entropy
        _, chosen_obj = min(entropy_list)

        # Pick the highest-likelihood valid name for this object
        candidates = [
            (name, prob)
            for name, prob in dist_per_obj[chosen_obj]
            if name_budget[name] > 0
        ]
        if not candidates:
            raise ValueError(f"No valid candidates left for object {chosen_obj}")

        best_name = max(candidates, key=lambda x: x[1])[0]

        assigned[chosen_obj] = best_name
        name_budget[best_name] -= 1

    return assigned

def compute_entropy_from_logits(valid_logits):
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

def assign_entities_with_dynamic_softmax(entity_list, predicted_logits):
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
        logits_per_obj[obj_id].append((name, logit))

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
            entropy, softmax_probs = compute_entropy_from_logits(valid_logits)
            entropy_list.append((entropy, obj_id))
            prob_per_obj[obj_id] = softmax_probs

        # Choose object with lowest entropy
        _, chosen_obj = min(entropy_list)
        softmax_probs = prob_per_obj[chosen_obj]

        # Pick the highest-softmax-prob name that's still valid
        available_names = [(name, prob) for name, prob in softmax_probs.items() if name_budget[name] > 0]
        if not available_names:
            raise ValueError(f"No valid candidates left for object {chosen_obj}")

        best_name = max(available_names, key=lambda x: x[1])[0]

        assigned[chosen_obj] = best_name
        name_budget[best_name] -= 1

    return assigned

if __name__ == "__main__":
    
    entity_list = ['purple cylinder', 'black cylinder', 'lime container', 'blue container', 'white triangular prism', 'black cylinder']
    color_list = [i.split(' ')[0] for i in entity_list]
    shape_list = [' '.join(i.split(' ')[1:]) for i in entity_list]
    
    predicted_probs = entities
    res = assign_entities_with_dynamic_softmax(entity_list, predicted_probs)

    predicted_colors = {(oid, color): p for (_, oid, color), p in colors.items()}
    res_color = assign_entities_with_dynamic_softmax(color_list, predicted_colors)
    res_shape = assign_entities_with_dynamic_softmax(shape_list, shapes)
    
    res_color = assign_entities_with_entropy(color_list, predicted_colors)
    res_shape = assign_entities_with_entropy(shape_list, shapes)
    res = assign_entities_with_entropy(entity_list, predicted_probs)
    print(res)