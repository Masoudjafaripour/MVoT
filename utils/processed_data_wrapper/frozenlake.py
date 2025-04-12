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
"""FrozenLake dataset for MVoT (toy version)"""

import json
import random
import os
import math
import ast
import shutil

import datasets
import string

from PIL import Image


_CITATION = """
@misc{brockman2016openaigym,
      title={OpenAI Gym}, 
      author={Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba},
      year={2016},
      eprint={1606.01540},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1606.01540}, 
}
"""

_DESCRIPTION = """
Frozen lake involves crossing a frozen lake from start to goal without falling into any holes 
by walking over the frozen lake. The player may not always move in the intended direction 
due to the slippery nature of the frozen lake.
"""

_HOMEPAGE = "https://gymnasium.farama.org/environments/toy_text/frozen_lake/"

_LICENSE = "CC BY 4.0"

_DATA_DIR = r"frozenlake"

_URLS = {
    "data_dir": _DATA_DIR,
}

SINGLE_STEP_VISUALIZATION_INSTRUCTION = {
    "frozenlake_simulation": "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
}
LONG_HORIZON_VISUALIZATION_INSTRUCTION = {
    "frozenlake_simulation": "<INIT_STATE>\nResponse: <ACTION_HISTORY>"
}
REAL_GOAL_INSTRUCTION = {
    "frozenlake_simulation": "Task: FrozenLake\nDetermine whether the agent (elf character) can safely reach the gift following the action sequence without falling into the holes. If not, identify the failure reason. The definitions of the actions are as below. \n* Go up/left/down/right: move one grid space in the absolute up/left/down/right direction. \nReturn A, B or C. \nFull Action Sequence: <ACTION_SEQ>\nA. Action Success. \nB. Action Failed: Fall into the Hole. \nC. Action Failed: Agent Safe but Fail to Reach Destination. \n"
}

ACTION_DICT = {
    0: "left",
    1: "down",
    2: "right",
    3: "up"
}

class FrozenLakeConfig(datasets.BuilderConfig):
    """BuilderConfig for FrozenLake."""

    def __init__(self, tasks, modes, data_dir, **kwargs):
        """BuilderConfig for FrozenLake.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FrozenLakeConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir


class FrozenLake(datasets.GeneratorBasedBuilder):
    """FrozenLake dataset."""

    BUILDER_CONFIG_CLASS = FrozenLakeConfig
    BUILDER_CONFIGS = [
        FrozenLakeConfig(
            name="processed_frozenlake",
            version=datasets.Version("0.0.0"),
            description=_DESCRIPTION,
            tasks=["simulation"],
            modes=["single_step_visualization", "action_reasoning"],
            data_dir="data_samples"
        )
    ]

    DEFAULT_CONFIG_NAME = "processed_frozenlake"

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
        data_dirs.append(os.path.join(data_dir, downloaded_files['data_dir']))     # should be named "frozenlake"

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

            json_files = [os.path.join(sub_dir, i) for sub_dir in all_datasets for i in os.listdir(os.path.join(data_dir, sub_dir)) if i.endswith("data.json")]

            train_data = []
            dev_data = []
            for json_f in json_files:
                with open(os.path.join(data_dir, json_f)) as f:
                    data = json.load(f)
                    total_env_num = len(data)

                    # FIXME: the training environments are set to the same as dev/test environment because this is just a toy dataset for debugging
                    # To use the formal implementation to avoid data contamination, uncomment the following code blocks
                    
                    # train_env_num = int(total_env_num * 0.8)

                    # for i in range(total_env_num):
                    #     if str(i) in data:
                    #         env_data = data[str(i)]
                    #         if i < train_env_num:
                    #             train_data += flatten_env_data(env_data, env_id=i)
                    #         else:
                    #             dev_data += flatten_env_data(env_data, env_id=i)
                    
                    for i in range(total_env_num):
                        if str(i) in data:
                            env_data = data[str(i)]
                            train_data += flatten_env_data(env_data, env_id=i)
                            dev_data += flatten_env_data(env_data, env_id=i)
                
            if split in ['train']:
                all_data += [{**item, "task": "frozenlake_simulation"} for item in train_data]
            else:
                all_data += [{**item, "task": "frozenlake_simulation"} for item in dev_data]
        
        data_idx = 0
        for data_item in all_data:
            interleaved_data_list = get_interleaved_data(
                data_item,
                mode=modes
            )
            
            for item in interleaved_data_list:
                action_list = data_item['action_list']
                additional_action_list = data_item['additional_actions']
                return_info = {
                    'idx': data_idx,
                    "input_text": item['input_text'].replace("<ACTION_SEQ>", "".join([f"Go {i.split('-')[0]}. " for i in action_list+additional_action_list])),
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

def index_to_coordinates(index, n):
    """Convert a 1-based index to 2D coordinates (row, col) in an n x n grid."""
    row = index // n
    col = index % n
    return [row, col]

def coordinates_to_index(row, col, n):
    """Convert 2D coordinates (row, col) to a 1-based index in an n x n grid."""
    return row * n + col

def flatten_env_data(env_dict, env_id):
    env_desc = env_dict['env_desc']
    return [
        {
            "action_list": [f"{ACTION_DICT[v]}-1" for v in env_dict["actions"][d_i]],
            "path_locs": [index_to_coordinates(v, int(math.sqrt(len(env_desc)))) for v in env_dict['states'][d_i]],
            "labels": env_dict["rewards"][d_i],
            "data_id": env_dict['data_id'][d_i],
            "env_desc": env_desc,
            "env_id": env_id,
            "grid_num": int(math.sqrt(len(env_desc))),
            "exec_state": env_dict['exec_states'][d_i],
            "additional_actions": [f"{ACTION_DICT[v]}-1" for v in env_dict["additional_actions"][d_i]]
        }
        for d_i in range(len(env_dict['data_id'])) if len(env_dict["actions"][d_i]) > 0
    ]

def get_interleaved_data(data_item, mode=["single_step_visualization", "action_reasoning"]):
    interleaved_data = []

    action_list = data_item['action_list']
    pos_list = data_item['path_locs']
    all_images = [os.path.join(f"frozenlake/level{data_item['grid_num']}/{data_item['data_id']}", f"{pos_idx}.png") for pos_idx in range(len(action_list))]

    if data_item['task'].endswith("simulation"):
        all_images = all_images + [os.path.join(f"frozenlake/level{data_item['grid_num']}/{data_item['data_id']}", f"{len(action_list)}.png")]
        all_actions = ["input_img"] + action_list
        all_pos = pos_list
        task = data_item['task']
    else:
        raise ValueError("Task not found. Should be one of [simulation]")
    
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
            input_image_paths = [os.path.join(_DATA_DIR_PREFIX, image_batch[0]), os.path.join(_DATA_DIR_PREFIX, image_batch[-2])] if len(image_batch) > 2 else [os.path.join(_DATA_DIR_PREFIX, image_batch[0])]
            
            label_image_paths = [os.path.join(_DATA_DIR_PREFIX, img_path) for img_path in image_batch[-1:]]

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
                "task": task, # if batch_idx != (len(image_batches) - 1) else f"<image>. Action sequence stopped. The answer is {get_answer(data_item)}. "
                "input_text": input_text,
                "label_text": "<image>",
                "input_imgs": input_imgs,
                "input_img_paths": input_image_paths,
                "label_imgs": label_imgs,
                "label_img_paths": label_image_paths,
                'train_task': "single_step_visualization",
                'coords': pos_batch,
                "exec_state": data_item['exec_state']
            }
            interleaved_data.append(return_info)
    
    if "action_reasoning" in mode:
        image_batches = [all_images[:1]] + [all_images[:1] + all_images[i:i+1] for i in range(1, len(all_images))]
        pil_image_batches = [all_pil_images[:1]] + [all_pil_images[:1] + all_pil_images[i:i+1] for i in range(1, len(all_pil_images))]
        action_batches = [all_actions[0:2]] + [all_actions[0:i+2] for i in range(1, len(all_actions))]

        pos_batches = [all_pos[0:2]] + [all_pos[0:i+2] for i in range(1, len(all_pos))]

        for batch_idx, (image_batch, pil_image_batch, action_batch, pos_batch) in enumerate(zip(image_batches, pil_image_batches, action_batches, pos_batches)):
            input_image_paths = [os.path.join(_DATA_DIR_PREFIX, image_batch[0]), os.path.join(_DATA_DIR_PREFIX, image_batch[-1])] if len(image_batch) != 1 else [os.path.join(_DATA_DIR_PREFIX, image_batch[0])]
            label_image_paths = []

            texts = get_text_from_actions(action_batch)
            input_texts = texts[:-1] if batch_idx != (len(image_batches) - 1) else texts
            output_texts = texts[-1:] if batch_idx != (len(image_batches) - 1) else [f"Action sequence stopped. The answer is {get_answer(data_item)}. "]

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
                "coords": pos_batch,
                "exec_state": data_item['exec_state']
            }
            interleaved_data.append(return_info)

    return interleaved_data

def get_text_from_actions(action_list):
    text_list = []
    for action in action_list:
        if action == "input_img":
            text_list.append("Initial State: ")
        elif action == "start":
            text_list.append("Start Point. ")
        else:
            meta_action = action.split("-")
            text_list.append(f"Go {meta_action[0]}. ")
    return text_list

def get_answer(data_item):
    if data_item['exec_state'] == "success":
        output_text = "A"
    elif data_item['exec_state'] == "terminated":
        output_text = "B"
    elif data_item['exec_state'] == "truncated":
        output_text = "C"
    return output_text