from utils import globals
import numpy as np
from PIL import Image
from PIL import Image
import os
import pandas as pd
import ast
from datetime import datetime
from moviepy.editor import ImageSequenceClip



'''
    Assign a green color shade based on the age value.
    This function is used in the 'animate_cell_age' function to create the animation.
    The assigned color represents the age of a cell, where a darker shade is used for recently born cells
    and a lighter shade is used as the cell approaches the maximum age (amax).
    Note: Currently, the function is only applied to alive ('a') cells.
'''
def assign_color(max_age, age):
    normalized_age = age / max_age  # Normalize age to range [0, 1]
    green_value = int(normalized_age * 255)  # Scale normalized age to range [0, 255]
    return (0, green_value, 0)


def create_cellstate_images(main_dir, cell_states, cell_ages, amax):
    colored_cell_state = []
    for r in range(len(cell_states)):
        cell_states_arr = cell_states[r]
        age = cell_ages[r]
        cell_state_color_palette = np.zeros(
            (cell_states_arr.shape[0], cell_states_arr.shape[1], 3), dtype=np.uint8)
        for i in range(cell_states_arr.shape[0]):
            for j in range(cell_states_arr.shape[1]):
                if(cell_states_arr[i][j] == 'q'):
                    cell_state_color_palette[i][j] = globals.QUIESCENT_COLOR
                elif(cell_states_arr[i][j] == 'd'): 
                    cell_state_color_palette[i][j] = globals.DECAY_COLOR
                else: 
                    cell_state_color_palette[i][j] = assign_color(amax, age[i][j])
        colored_cell_state.append(cell_state_color_palette)

    images = [Image.fromarray(state) for state in colored_cell_state]
    # save_images(main_dir, 'cell_states_images', images)
    # create_image_collage(main_dir, 'cell_states_images', images)
    return images

def create_cellrule_images(main_dir, rules):
    
    color_palette = []
    rules_colors_map_df = pd.read_csv('utils/rules_colors_map.csv')
    rules_colors_map_df['colors'] = rules_colors_map_df['colors'].apply(ast.literal_eval)
    for i in range(len(rules)):
        rule = rules[i]
        datetime1 = datetime.now()
        rule_color = np.empty(rule.shape, dtype=object)
        unique_rules = list(np.unique(rule))
        subset_rule_color_df = rules_colors_map_df[rules_colors_map_df['rules'].isin(unique_rules)]
        rule_color_dict = dict(zip(subset_rule_color_df['rules'], subset_rule_color_df['colors']))
        for u in unique_rules:
            if u == 'nan':
                rule_color[rule == 'nan'] = '[255,255,255]'
            else:
                color = rule_color_dict[u]
                rule_color[rule == u] = str(color)    
        
        vectorized_eval = np.frompyfunc(safe_eval, 1, 1)
        rule_color = vectorized_eval(rule_color)
        rule_color = np.array(rule_color.tolist()).astype(np.uint8)
        color_palette.append(rule_color)
        print('time taken for a loop : ', datetime.now() - datetime1)

    images = [Image.fromarray(state) for state in color_palette]
    # save_images(main_dir, 'cell_genomes_images', images)
    # create_image_collage(main_dir, 'cell_genomes_images', images,)
    return images

def create_gridstate_images(main_dir, grids, cell_states):
    colored_grid = []
    for g in range(len(grids)):
        grid = grids[g]
        cell_state = cell_states[g]
        color_palette = np.zeros(
            (grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if(cell_state[i][j] == 'q'):
                    color_palette[i][j] = [255, 255, 255]
                elif(grid[i][j] == 0):
                    color_palette[i][j] = [255, 0, 0]  #red
                else:
                    color_palette[i][j] = [0, 255, 0]    #green
        colored_grid.append(color_palette)

    images = [Image.fromarray(state) for state in colored_grid]
    # save_images(main_dir, 'grid_states_images', images)
    # create_image_collage(main_dir, 'grid_states_images', images)
    return images

def create_movie(imagelist: list, dir: str):
    imagelist = [np.array(img) for img in imagelist]
    fps =globals.FPS if globals.FPS else 250
    clip = ImageSequenceClip(imagelist, fps=fps)
    clip.write_videofile(dir , codec='libx264')


def create_image_collage_of_each_ca_properties(main_dir: str, imagelist: list):
    image_collages = []
    image_width, image_height = imagelist[0][0].size
    scaling_factor = len(imagelist[0])
    image_gap = int(image_width/7)
    image_width = scaling_factor * (image_width + image_gap)
    
    for i in range(len(imagelist)):
        image_tuple = imagelist[i]
        collage = Image.new('RGB', (image_width, image_height))
        x, y = int(image_gap/2), 0

        for j in range(len(image_tuple)):
            img = image_tuple[j]
            ind_img_width = img.width
            img = img.resize((ind_img_width, image_height))
            collage.paste(img, (x, y))
            x += img.width + image_gap
        image_collages.append(collage)

    save_images(main_dir, 'properities_collage', image_collages)
    return image_collages


def create_image_collage(dir,sub_dir, images):


    image_width, image_height = images[0].size

    output_dir = f'{dir}/{sub_dir}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scaling_factor = len(images)
    
    image_gap = int(image_width/7)
    image_width = scaling_factor * (image_width + image_gap)
    collage = Image.new('RGB', (image_width, image_height))

    # Initialize coordinates for placing images in the collage
    x, y = 0, 0

    for image in images:
        # Resize the image to fit the collage dimensions
        ind_img_width = image.width
        image = image.resize((ind_img_width, image_height))
        collage.paste(image, (x, y))
        x += image.width + image_gap

    # Save the collage
    collage.save(f'{output_dir}/{sub_dir}_collage.png')


def save_images(dir,sub_dir, images):
    output_dir = f'{dir}/{sub_dir}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, image in enumerate(images, start=1):
        output_image_path = os.path.join(output_dir, f'image_{i}.png')
        image.save(output_image_path)


def safe_eval(element):
    return ast.literal_eval(element)

'''
    This function orchestrates the creation of the animation. 
    It takes a DataFrame (df) containing cell rules, cell states, cell ages, and the grid, 
    as well as the rule_type parameter to determine the type of rules used in the animation.
'''
def visual_result(properties_dict: dict, param: dict):
    print("Creating Animation")
    dir = param['animation_dir']
    cell_rules = properties_dict['cell_rules']
    cell_states = properties_dict['cell_states']
    cell_age = properties_dict['cell_ages']
    grid = properties_dict['grid_states']

    cellstate_dir = f'{dir}/{param["cellstate_filename"]}'
    gridstate_dir= f'{dir}/{param["gridstate_filename"]}'
    cellrule_dir = f'{dir}/{param["cellrule_filename"]}'

    print("Start - Cell State Animation Creation")
    cellstate_images = create_cellstate_images(dir, cell_states, cell_age, param['amax'])
    # create_movie(cellstate_images, cellstate_dir)
    print("Complete - Cell State Animation Creation")
   
    print("Start - Cell Grid Animation Creation")
    grid_images = create_gridstate_images(dir, grid, cell_states)
    # create_movie(grid_images, gridstate_dir)
    print("Complete - Cell Grid Animation Creation")

    print("Start - Cell Rule Animation Creation")
    cellrule_images = create_cellrule_images(dir,cell_rules)
    # create_movie(cellrule_images, cellrule_dir)
    print("Complete - Cell Rule Animation Creation")

    image_tuples = [list(t) for t in zip(cellrule_images, cellstate_images, grid_images)]
    image_collages = create_image_collage_of_each_ca_properties(dir, image_tuples)

    print("Start - Collage Creation")
    collage_dir = f'{dir}/animation_collage.mp4'
    create_movie(image_collages, collage_dir)
    print("Complete - Collage Creation")

    print("COMPLETED - ANIMATION CREATION")

    
    return {'cell_states': cellstate_dir, 'cell_rules': gridstate_dir, 'grid_states': cellrule_dir}
