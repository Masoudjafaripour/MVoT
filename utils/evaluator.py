import re

import numpy as np

from evaluate import load

class VisualizationEvaluator():
    def __init__(self, **kwargs):
        pass

    def evaluate(self, text_preds, gold_data, sketch_preds, section, finish):
        if "train_task" in gold_data[0].keys():
            train_with_image = True
        else:
            train_with_image = False

        gold_coords = [i['coords'] for i in gold_data]
        gold_tasks = [i['task'] for i in gold_data]
        gold_text = [i['label_text'] for i in gold_data]
        maze_sizes = [i['maze_size'] for i in gold_data]
        if train_with_image:
            gold_train_task = [i['train_task'] for i in gold_data]
        else:
            gold_train_task = [None] * len(gold_data)

        text_preds = [i.split('<reserved08706>')[0] for i in text_preds]

        simulation_match = r'The answer is (\w)\.'

        result_dict = {}

        acc_score = 0
        sim_acc_score = 0
        
        perceptual_acc = 0
        perceptual_action_acc = 0
        perceptual_redundant = 0

        task_data_num = 0
        sim_data_num = 0
        perceptual_data_num = 0

        for task, text, output, train_task, coords, maze_size, sketch in zip(gold_tasks, gold_text, text_preds, gold_train_task, gold_coords, maze_sizes, sketch_preds):
            if task not in result_dict.keys():
                if "simulation" in task:
                    result_dict[task] = {"task_data_num": 0, "task_acc": 0}
                elif "path_reading" in task:
                    result_dict[task] = {"task_data_num": 0, "prefix_acc": 0, "overall_portion_acc": 0, "strict_acc": 0}
            
            if "simulation" in task:
                label = re.findall(simulation_match, text)
                pred = re.findall(simulation_match, output)

                if label:
                    sim_data_num += 1
                    min_num = min(len(label), len(pred))
                    sim_acc_score += sum([i==j for i, j in zip(label[:min_num], pred[:min_num])]) / len(label)
                    
                    result_dict[task]['task_data_num'] += 1
                    min_num = min(len(label), len(pred))
                    result_dict[task]['task_acc'] += sum([i==j for i, j in zip(label[:min_num], pred[:min_num])]) / len(label)

                # for perceptual accuracy with generated sketch
                if train_with_image:
                    if "visualization" in train_task and sketch is not None and task == "maze_simulation":
                        perceptual_item_acc = 0
                        for coord in coords:
                            pixel_coords = get_pixel_location(
                                maze_size=maze_size,
                                coord=coord,
                                img_size=362
                            )
                            bias = (61, 147)    # manually set to parse the maze
                            pixel_coords = (pixel_coords[0] + bias[0], pixel_coords[1] + bias[1])
                            down_right_coords = (pixel_coords[0] + int(362 / maze_size), pixel_coords[1] + int(362 / maze_size))
                            # resize to 512x512
                            pixel_coords = (int(pixel_coords[0] / 480 * 512), int(pixel_coords[1] / 640 * 512))
                            down_right_coords = (int(down_right_coords[0] / 480 * 512), int(down_right_coords[1] / 640 * 512))

                            perceptual_area = sketch[
                                0, 
                                :, 
                                pixel_coords[0]:down_right_coords[0], 
                                pixel_coords[1]:down_right_coords[1]
                                ]
                            # assert if there is any color in this area
                            perceptual_indicator = is_similar_to_red(perceptual_area.squeeze()).detach().cpu().numpy()
                            perceptual_item_acc += 1 if np.sum(perceptual_indicator) / perceptual_indicator.size > 0.025 else 0

                            # mark this area to pure black
                            sketch[
                                0, 
                                :, 
                                pixel_coords[0]:down_right_coords[0], 
                                pixel_coords[1]:down_right_coords[1]
                                ] = 0
                        
                        perceptual_redundant_indicator = is_similar_to_red(sketch.squeeze()).detach().cpu().numpy()
                            
                        perceptual_acc += perceptual_item_acc / len(coords)
                        perceptual_action_acc += 1 if np.sum(perceptual_indicator) / perceptual_indicator.size > 0.025 else 0
                        perceptual_redundant += 1 if np.sum(perceptual_redundant_indicator) / perceptual_redundant_indicator.size > 0.0001 else 0
                        perceptual_data_num += 1

                    if sketch is not None and task == "minibehavior_simulation":
                        perceptual_item_acc = 0
                        img_size = 512

                        coord = coords[-1][::-1]
                        pixel_coords = get_pixel_location(
                            maze_size=maze_size,
                            coord=coord,
                            img_size=img_size
                        )
                        bias = (0, 0)
                        pixel_coords = (pixel_coords[0] + bias[0], pixel_coords[1] + bias[1])
                        down_right_coords = (pixel_coords[0] + int(img_size / maze_size), pixel_coords[1] + int(img_size / maze_size))

                        perceptual_area = sketch[
                            0, 
                            :, 
                            pixel_coords[0]:down_right_coords[0], 
                            pixel_coords[1]:down_right_coords[1]
                            ]
                        # assert if there is any color in this area
                        perceptual_indicator = is_similar_to_red(perceptual_area.squeeze(), 180).detach().cpu().numpy()
                        perceptual_item_acc += 1 if np.sum(perceptual_indicator) / perceptual_indicator.size > 0.025 else 0

                        # mark this area to pure black
                        sketch[
                            0, 
                            :, 
                            pixel_coords[0]:down_right_coords[0], 
                            pixel_coords[1]:down_right_coords[1]
                            ] = 0
                        
                        perceptual_redundant_indicator = is_similar_to_red(sketch.squeeze(), 180).detach().cpu().numpy()
                            
                        perceptual_acc += perceptual_item_acc
                        perceptual_action_acc += 1 if np.sum(perceptual_indicator) / perceptual_indicator.size > 0.025 else 0
                        perceptual_redundant += 1 if np.sum(perceptual_redundant_indicator) / perceptual_redundant_indicator.size > 0.0001 else 0
                        perceptual_data_num += 1
                    
            if label:
                task_data_num += 1
                min_num = min(len(label), len(pred))
                acc_score += sum([i==j for i, j in zip(label[:min_num], pred[:min_num])]) / len(label)
        
        perceptual_acc_score = perceptual_acc / perceptual_data_num if perceptual_data_num != 0 else 0
        perceptual_action_acc_score = perceptual_action_acc / perceptual_data_num if perceptual_data_num != 0 else 0
        perceptual_redundant_score = perceptual_redundant / perceptual_data_num if perceptual_data_num != 0 else 0

        all_task_results = {f"{task_name}_{metric_name}": task_metrics[metric_name] / task_metrics['task_data_num'] if task_metrics['task_data_num'] != 0 else 0 for task_name, task_metrics in result_dict.items() for metric_name in task_metrics.keys() if metric_name not in ['task_data_num']}

        simulation_vis_results = {
            "simulation_visualization_overall_acc": perceptual_acc_score,
            "simulation_visualization_next_action_acc": perceptual_action_acc_score,
            "simulation_visualization_redundancy": perceptual_redundant_score
        }
        
        final_results = {
            "overall_task_acc": acc_score / task_data_num if task_data_num != 0 else 0,
            **all_task_results,
            **simulation_vis_results
        }
        return final_results


def get_pixel_location(maze_size, coord, img_size=512):
    """
    Calculate the pixel location of a given maze coordinate.
    
    Parameters:
    maze_size (int): Size of the maze (e.g., 3 for 3x3, 4 for 4x4, etc.)
    coord (tuple): The maze coordinates (row, col) as (x, y).
    img_size (int): The size of the image in pixels. Default is 512x512.
    
    Returns:
    tuple: The top-left corner of the cell in pixel coordinates.
    """
    # Ensure the coordinate is valid for the given maze size
    if coord[0] >= maze_size or coord[1] >= maze_size:
        raise ValueError(f"Coordinate {coord} is out of bounds for a {maze_size}x{maze_size} maze.")
    
    # Calculate the size of each cell in pixels
    cell_size = img_size // maze_size
    
    # Calculate pixel position (top-left corner of the cell)
    pixel_x = coord[0] * cell_size
    pixel_y = coord[1] * cell_size
    
    return (pixel_x, pixel_y)

def is_similar_to_red(image_array, threshold=120):
    """Check if a pixel is similar to red based on a threshold."""
    r, g, b = image_array[0, :, :], image_array[1, :, :], image_array[2, :, :]
    # r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    red_like_mask = (r > threshold) & (g < threshold // 1.1) & (b < threshold // 1.1)
    return red_like_mask

def is_similar_to_blue(image_array, threshold=120):
    """Check if a pixel is similar to blue based on a threshold."""
    r, g, b = image_array[0, :, :], image_array[1, :, :], image_array[2, :, :]
    blue_like_mask = (b > threshold) & (g < threshold // 1.1) & (r < threshold // 1.1)
    return blue_like_mask

def is_not_similar_to_white_or_light_blue(image_array, white_threshold=200, blue_threshold=200):
    """Check if pixels are not similar to white or light blue based on thresholds."""
    r, g, b = image_array[0, :, :], image_array[1, :, :], image_array[2, :, :]
    # r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    # White-like mask: all channels are high
    white_like_mask = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)
    # Light blue-like mask: blue is dominant, red and green are moderate
    light_blue_like_mask = (b > blue_threshold) & (g > blue_threshold // 1.5) & (r < blue_threshold // 1.5)
    # Combined mask for pixels similar to white or light blue
    similar_mask = white_like_mask | light_blue_like_mask
    # Invert to get pixels not similar to white or light blue
    not_similar_mask = ~similar_mask
    return not_similar_mask