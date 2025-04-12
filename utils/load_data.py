import os
import torch
import json

from datasets import load_dataset, concatenate_datasets

from utils.tokenized_dataset import AnoleTokenizedDataset
from utils.interleaved_tokenized_dataset import InterleaveAnoleTokenizedDataset

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

def load_data(dataset, data_dir):
    data_list = []
    if 'interleaved_maze' in dataset:
        data = load_dataset(
            "utils/processed_data_wrapper/interleaved_maze.py", 
            tasks=['simulation'], 
            modes=['single_step_visualization', 'action_reasoning'], 
            data_dir=data_dir
        )
        print(f"Interleaved Maze: {len(data['train'])}")
        data_list.append(data)
    if 'frozenlake' in dataset:
        data = load_dataset(
            "utils/processed_data_wrapper/frozenlake.py", 
            tasks=['simulation'], 
            modes=['single_step_visualization', 'action_reasoning'], 
            data_dir=data_dir
        )
        print(f"FrozenLake: {len(data['train'])}")
        data_list.append(data)

    concatenate_data = dict()
    for k in data.keys():
        if k in ['train']:
            concatenate_data[k] = concatenate_datasets([i[k] for i in data_list])
        else:
            concatenate_data[k] = concatenate_datasets([i[k].shuffle(seed=42).select(range(800)) for i in data_list])
    return concatenate_data

def tokenize_dataset(train_split, eval_split, test_split, model, processor, **kwargs):
    tokenized_data = dict()

    data_name = kwargs.pop("data_name")

    max_source_length = 2600
    print(f"Max source length: {max_source_length}")

    max_target_length = 1300
    print(f"Max target length: {max_target_length}")

    if not kwargs["interleave"]:
        tokenized_dataset_type = AnoleTokenizedDataset
    else:
        tokenized_dataset_type = InterleaveAnoleTokenizedDataset

    if train_split:
        tokenized_train = tokenized_dataset_type(
            dataset=train_split,
            split='train',
            model=model,
            processor=processor,
            input_max_length=max_source_length, 
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['train'] = tokenized_train
    if eval_split:
        tokenized_eval = tokenized_dataset_type(
            dataset=eval_split,
            split='eval',
            model=model,
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['eval'] = tokenized_eval
    if test_split:
        tokenized_test = tokenized_dataset_type(
            dataset=test_split,
            split='test',
            model=model,
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['test'] = tokenized_test
    return tokenized_data, max_source_length, max_target_length


def get_image_token_num(model, processor, resolution):
    if hasattr(processor, 'image_seq_length'):
        return processor.image_seq_length
    elif hasattr(model, get_image_token_num):
        return model.get_image_token_num(resolution=resolution)
    else:
        raise NotImplementedError("Either model should have the get_image_token_num method or processor should have the iamge_seq_length property. ")