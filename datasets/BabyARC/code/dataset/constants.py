import os

## Constants
position_tag_upper = "upper"
position_tag_lower = "lower"
position_tag_left = "left"
position_tag_right = "right"
position_tag_none = "none"

## File locations

arc_data_dirname = os.path.join("ARC/data")

arc_data_train_path = os.path.join(arc_data_dirname, "training")

arc_data_eval_path = os.path.join(arc_data_dirname, "evaluation")

tmp_data_dirname = os.path.join("/concept_env/datasets/")