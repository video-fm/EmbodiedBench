# Task-Based Folder Structure

This document describes the organized task-based folder structure for EmbodiedBench evaluators and planners.

## Overview

The codebase is now organized by task type, with three experimental variants for each task:
- **base**: Baseline implementation
- **gd**: Grounding DINO integration
- **esca**: Enhanced Scene Context Awareness (LASER + Grounding DINO)

## Folder Structure

```
embodiedbench/
├── evaluator/
│   ├── alfred/
│   │   ├── base.py      (✓ Implemented)
│   │   ├── gd.py        (✓ Implemented)
│   │   └── esca.py      (✓ Implemented)
│   ├── navigation/
│   │   ├── base.py      (✓ Implemented)
│   │   ├── gd.py        (✓ Implemented)
│   │   └── esca.py      (✓ Implemented)
│   ├── habitat/
│   │   ├── base.py      (✓ Implemented)
│   │   ├── gd.py        (⚠ Placeholder)
│   │   └── esca.py      (⚠ Placeholder)
│   └── manipulation/
│       ├── base.py      (✓ Implemented)
│       ├── vla.py       (✓ Implemented - VLA variant)
│       ├── gd.py        (⚠ Placeholder)
│       └── esca.py      (⚠ Placeholder)
│
└── planner/
    ├── alfred/
    │   ├── base.py      (✓ Implemented)
    │   ├── gd.py        (✓ Implemented)
    │   └── esca.py      (✓ Implemented)
    ├── navigation/
    │   ├── base.py      (✓ Implemented)
    │   ├── gd.py        (✓ Implemented)
    │   └── esca.py      (✓ Implemented)
    ├── habitat/
    │   ├── base.py      (⚠ Placeholder)
    │   ├── gd.py        (⚠ Placeholder)
    │   └── esca.py      (⚠ Placeholder)
    └── manipulation/
        ├── base.py      (✓ Implemented)
        ├── vla.py       (✓ Implemented - VLA variant)
        ├── gd.py        (⚠ Placeholder)
        └── esca.py      (⚠ Placeholder)
```

## Task Descriptions

### Alfred
**Purpose**: Action Learning from Realistic Demonstrations  
**Implemented Variants**: base, gd, esca  
**Evaluator**: Language-guided household tasks in AI2-THOR  
**Planner**: VLM-based planning with scene graph support

### Navigation
**Purpose**: Object-goal navigation tasks  
**Implemented Variants**: base, gd, esca  
**Evaluator**: Navigate to target objects in various scenes  
**Planner**: Vision-language navigation planning

### Habitat
**Purpose**: Embodied AI in photorealistic environments  
**Implemented Variants**: base only (gd and esca pending)  
**Evaluator**: Habitat simulator-based tasks  
**Planner**: TODO - Habitat-specific planning

### Manipulation
**Purpose**: Robot manipulation tasks  
**Implemented Variants**: base, vla (gd and esca pending)  
**Evaluator**: Tabletop manipulation in simulation  
**Planner**: Manipulation-specific planning with VLA support

## Usage

### Import Examples

```python
# Old style (still works, but deprecated)
from embodiedbench.evaluator.eb_alfred_evaluator import EB_AlfredEvaluator

# New organized style
from embodiedbench.evaluator.alfred.base import EB_AlfredEvaluator
from embodiedbench.evaluator.alfred.gd import EB_AlfredEvaluator as AlfredGDEvaluator
from embodiedbench.evaluator.alfred.esca import EB_AlfredEvaluator as AlfredESCAEvaluator

# Using __init__.py imports
from embodiedbench.evaluator.alfred import AlfredBaseEvaluator
from embodiedbench.planner.navigation import NavigationBasePlanner
```

### Running Experiments

```bash
# Run base Alfred evaluator
python -m embodiedbench.evaluator.alfred.base

# Run GD-enhanced Navigation evaluator
python -m embodiedbench.evaluator.navigation.gd

# Run ESCA Alfred evaluator
python -m embodiedbench.evaluator.alfred.esca
```

## Backward Compatibility

The original files remain in the root directories for backward compatibility:
- `embodiedbench/evaluator/eb_alfred_evaluator.py` → Same as `alfred/base.py`
- `embodiedbench/evaluator/eb_alfred_evaluator_gd.py` → Same as `alfred/gd.py`
- `embodiedbench/evaluator/eb_alfred_evaluator_esca.py` → Same as `alfred/esca.py`
- etc.

However, new development should use the task-organized structure.

## Implementation Status

### Fully Implemented
- ✅ Alfred: base, gd, esca (evaluator + planner)
- ✅ Navigation: base, gd, esca (evaluator + planner)
- ✅ Habitat: base (evaluator only)
- ✅ Manipulation: base, vla (evaluator + planner)

### Placeholders (TODO)
- ⚠️ Habitat: gd, esca variants
- ⚠️ Habitat: all planners
- ⚠️ Manipulation: gd, esca variants

## Contributing

When adding new task variants:
1. Create the file in the appropriate task folder
2. Update the `__init__.py` to export the new class
3. Add backward compatibility link if needed
4. Update this README
5. Add tests for the new variant

## Migration Guide

To migrate existing code to the new structure:

1. **Update imports**:
   ```python
   # Old
   from embodiedbench.evaluator.eb_alfred_evaluator import EB_AlfredEvaluator
   
   # New
   from embodiedbench.evaluator.alfred.base import EB_AlfredEvaluator
   ```

2. **Update file references**: If you have hardcoded file paths, update them to the new structure

3. **Test**: Run your existing tests to ensure compatibility

## Future Work

- [ ] Implement Habitat planners (base, gd, esca)
- [ ] Implement Habitat GD and ESCA evaluators
- [ ] Implement Manipulation GD and ESCA variants
- [ ] Add unit tests for each task-variant combination
- [ ] Create unified configuration system for all variants

