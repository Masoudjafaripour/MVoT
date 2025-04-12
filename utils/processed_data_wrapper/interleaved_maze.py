# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Maze dataset for MVoT (toy version)"""

import json
import os
import ast
import shutil

import datasets
import string

from PIL import Image, ImageOps


_CITATION = """
@misc{maze-dataset,
    title={A Configurable Library for Generating and Manipulating Maze Datasets}, 
    author={Michael Igorevich Ivanitskiy and Rusheb Shah and Alex F. Spies and Tilman Räuker and Dan Valentine and Can Rager and Lucia Quirke and Chris Mathwin and Guillaume Corlouer and Cecilia Diniz Behn and Samy Wu Fung},
    year={2023},
    eprint={2309.10498},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={http://arxiv.org/abs/2309.10498}
}
"""

_DESCRIPTION = """
This package includes a variety of maze generation algorithms, including randomized depth first search, 
Wilson's algorithm for uniform spanning trees, and percolation. Datasets can be filtered to select mazes of a certain length or complexity, 
remove duplicates, and satisfy custom properties. A variety of output formats for visualization and training ML models are provided.
"""

_HOMEPAGE = "https://github.com/understanding-search/maze-dataset"

_LICENSE = "CC BY 4.0"

_DATA_DIR = r"maze"

_URLS = {
    "data_dir": _DATA_DIR,
}

SINGLE_STEP_VISUALIZATION_INSTRUCTION = {
    "maze_simulation": "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
}
LONG_HORIZON_VISUALIZATION_INSTRUCTION = {
    "maze_simulation": "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
}
REAL_GOAL_INSTRUCTION = {
    "maze_simulation": "Task: Maze Navigation Simulation\nDetermine the final destination (A, B, C or D) from the starting point (red point) following the action sequence. The definitions of the actions are as below. \n* Go up/left/down/right: move one grid space in the absolute up/left/down/right direction. \nFull Action Sequence: <ACTION_SEQ>\n"
}



class MazeConfig(datasets.BuilderConfig):
    """BuilderConfig for Maze."""

    def __init__(self, tasks, modes, data_dir, **kwargs):
        """BuilderConfig for Maze.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MazeConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir


class Maze(datasets.GeneratorBasedBuilder):
    """Maze dataset."""

    BUILDER_CONFIG_CLASS = MazeConfig
    BUILDER_CONFIGS = [
        MazeConfig(
            name="processed_maze",
            version=datasets.Version("0.0.0"),
            description=_DESCRIPTION,
            tasks=["simulation"],
            modes=["single_step_visualization", "action_reasoning"],
            data_dir="data_samples"
        )
    ]

    DEFAULT_CONFIG_NAME = "processed_maze"

    def _info(self):
        features = datasets.Features(
            {
                'idx': datasets.Value('int32'),
                "input_text": datasets.Value("string"),
                "input_imgs": datasets.Sequence(datasets.Image()),
                "label_text": datasets.Value("string"),
                "label_imgs": datasets.Sequence(datasets.Image()),
                "label_img_paths": datasets.Sequence(datasets.Value("string")),
                "input_img_paths": datasets.Sequence(datasets.Value("string")),
                'task': datasets.Value('string'), 
                'train_task': datasets.Value("string"),
                'coords': datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                'maze_size': datasets.Value("int32")
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = _URLS

        tasks = self.config.tasks
        modes = self.config.modes
        data_dir = self.config.data_dir

        global _DATA_DIR_PREFIX
        _DATA_DIR_PREFIX = data_dir

        data_dirs = []
        for task in tasks:
            data_dirs.append(os.path.join(data_dir, downloaded_files['data_dir']))
                  
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "train",
                    "modes": modes
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('dev'),
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "dev",
                    "modes": modes
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('test'),
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "test",
                    "modes": modes
                },
            ),
        ]

    def _generate_examples(self, data_dirs: list, split: str, modes: list):
        all_data = []
        for data_dir in data_dirs:
            all_datasets = os.listdir(data_dir)

            json_files = [i for i in all_datasets if i.endswith(".json")]

            for json_f in json_files:
                with open(os.path.join(data_dir, json_f)) as f:
                    data = json.load(f)
                    diff_data = remove_duplicates(data)

                    train_num = int(len(diff_data) * 0.8)
                
                # FIXME: the training environments are set to the same as dev/test environment because this is just a toy dataset for debugging
                # To use the formal implementation to avoid data contamination, uncomment the following code blocks
                    
                # if split in ['train']:
                #     all_data += [{**item, "task": data_dir} for item in diff_data[:train_num]]
                # else:
                #     all_data += [{**item, "task": data_dir} for item in diff_data[train_num:]]

                all_data += [{**item, "task": "maze_simulation"} for item in diff_data]
        
        data_idx = 0
        for data_item in all_data:
            interleaved_data_list = get_interleaved_data(
                data_item,
                mode=modes
            )
            
            for item in interleaved_data_list:
                return_info = {
                    'idx': data_idx,
                    "input_text": item['input_text'].replace("<ACTION_SEQ>", "".join([f"Go {i.split('-')[0]}. " for i in data_item['action_list']])),
                    "input_imgs": item["input_imgs"],
                    "label_text": item['label_text'],
                    "label_imgs": item['label_imgs'],
                    "label_img_paths": item['label_img_paths'],
                    "input_img_paths": item['input_img_paths'],
                    "task": item['task'],
                    "train_task": item['train_task'],
                    "coords": item['coords'],
                    "maze_size": data_item['grid_num']
                }
                yield data_idx, return_info
                data_idx += 1

def remove_duplicates(data_dict_list):
    past_paths = []
    unique_list = []
    for data_dict in data_dict_list:
        sublist = data_dict['path_locs']
        if sublist not in past_paths:
            past_paths.append(sublist)
            unique_list.append(data_dict)
    return unique_list

def get_interleaved_data(data_item, mode=["single_step_visualization", "action_reasoning"]):
    interleaved_data = []

    action_list = data_item['action_list']
    pos_list = data_item['path_locs']
    all_images = [data_item['file_prefix'] + f"_{pos_idx+1}.jpg" for pos_idx in range(len(action_list))]

    if data_item['task'].endswith("simulation"):
        all_images = all_images + [data_item['file_prefix'] + f"_{len(action_list)+1}.jpg"]
        all_actions = ["input_img"] + action_list
        all_pos = pos_list
        task = "maze_simulation"
    else:
        raise ValueError("Task not found. Should be within [simulation]")
    
    all_images = [os.path.join(_DATA_DIR_PREFIX, p) for p in all_images]
    try:
        all_pil_images = [Image.open(input_img_path).convert("RGB").resize((256, 256)) for input_img_path in all_images]
    except:
        return interleaved_data

    if "single_step_visualization" in mode:
        image_batches = [all_images[:2]] + [all_images[:1] + all_images[i:i+2] for i in range(1, len(all_images)-1)]
        pil_image_batches = [all_pil_images[:2]] + [all_pil_images[:1] + all_pil_images[i:i+2] for i in range(1, len(all_pil_images)-1)]
        action_batches = [all_actions[:2]] + [all_actions[0:i+2] for i in range(1, len(all_actions)-1)]

        pos_batches = [all_pos[:2]] + [all_pos[0:i+2] for i in range(1, len(all_pos)-1)]

        for batch_idx, (image_batch, pil_image_batch, action_batch, pos_batch) in enumerate(zip(image_batches, pil_image_batches, action_batches, pos_batches)):
            input_image_paths = [image_batch[0], image_batch[-2]] if len(image_batch) != 2 else [image_batch[0]]
            label_image_paths = image_batch[-1:]

            texts = get_text_from_actions(action_batch)
            input_texts = texts
            output_texts = ["<image>"]

            history_text = "".join(input_texts[1:-1]) + "<image>" + input_texts[-1] if len(input_texts) != 2 else input_texts[-1]
            init_state_text = input_texts[0] + "<image>"
            input_text = SINGLE_STEP_VISUALIZATION_INSTRUCTION[task].replace("<INIT_STATE>", init_state_text).replace("<ACTION_HISTORY>", history_text)
            input_text = REAL_GOAL_INSTRUCTION[task] + input_text

            input_imgs = pil_image_batch[:1] + pil_image_batch[-2:-1] if len(pil_image_batch) != 2 else pil_image_batch[:1]
            label_imgs = pil_image_batch[-1:]

            return_info = {
                "task": task,
                "input_text": input_text,
                "label_text": "<image>",
                "input_imgs": input_imgs,
                "input_img_paths": input_image_paths,
                "label_imgs": label_imgs,
                "label_img_paths": label_image_paths,
                'train_task': "single_step_visualization",
                'coords': pos_batch
            }
            interleaved_data.append(return_info)
    
    if "action_reasoning" in mode:
        image_batches = [all_images[:1]] + [all_images[:1] + all_images[i:i+1] for i in range(1, len(all_images))]
        pil_image_batches = [all_pil_images[:1]] + [all_pil_images[:1] + all_pil_images[i:i+1] for i in range(1, len(all_pil_images))]
        action_batches = [all_actions[0:2]] + [all_actions[0:i+2] for i in range(1, len(all_actions))]

        pos_batches = [all_pos[:2]] + [all_pos[0:i+2] for i in range(1, len(all_pos))]

        for batch_idx, (image_batch, pil_image_batch, action_batch, pos_batch) in enumerate(zip(image_batches, pil_image_batches, action_batches, pos_batches)):
            input_image_paths = [image_batch[0], image_batch[-1]] if len(image_batch) != 1 else [image_batch[0]]
            label_image_paths = []

            texts = get_text_from_actions(action_batch)
            input_texts = texts[:-1] if batch_idx != (len(image_batches) - 1) else texts
            output_texts = texts[-1:] if batch_idx != (len(image_batches) - 1) else [f"Action sequence finished. The answer is {get_answer(data_item)}. "]

            history_text = "".join(input_texts[1:]) + "<image>" if batch_idx != 0 else ""
            init_state_text = input_texts[0] + "<image>"
            input_text = LONG_HORIZON_VISUALIZATION_INSTRUCTION[task].replace("<INIT_STATE>", init_state_text).replace("<ACTION_HISTORY>", history_text)
            input_text = REAL_GOAL_INSTRUCTION[task] + input_text

            input_imgs = pil_image_batch[:1] + pil_image_batch[-1:] if batch_idx != 0 else pil_image_batch[:1]
            label_imgs = []

            return_info = {
                "task": task, 
                "input_text": input_text,
                "label_text": " ".join([o_t.strip() for o_t in output_texts]), 
                "input_imgs": input_imgs,
                "input_img_paths": input_image_paths,
                "label_imgs": label_imgs,
                "label_img_paths": label_image_paths,
                "train_task": "action_reasoning",
                "coords": pos_batch
            }
            interleaved_data.append(return_info)

    return interleaved_data

def get_text_from_actions(action_list):
    text_list = []
    for action in action_list:
        if action == "input_img":
            text_list.append("Initial maze: ")
        elif action == "start": 
            text_list.append("Start Point. ")
        else:
            meta_action = action.split("-")
            text_list.append(f"Go {meta_action[0]}. ")
    return text_list

def get_answer(data_item):
    if data_item['task'].endswith("simulation"):
        task = "simulation"

        options = [tuple(i) for i in data_item['candidate_locs']]
        destination_loc = tuple(data_item['path_locs'][-1])
        output_text = string.ascii_uppercase[options.index(destination_loc)]
    
    else:
        raise ValueError("Unsupported task. ")
    return output_text