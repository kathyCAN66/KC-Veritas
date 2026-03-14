# 2 types of objects: cups and bottles
# 2 colors: red and blue
# 2 positions: left and right (from the center)
# 2 sizes: big and small
# number of objects: range from 2 to 10
# 1 referring expression / scene (cup or bottle)

import os
import pandas as pd
import numpy as np
import csv
import random

# seed
random.seed(6)
np.random.seed(6)

def generate_scene():
    n = random.randint(2, 10)

    objects = []
    for _ in range(n):
        obj_type = random.choice(['cup', 'bottle'])
        color = random.choice(['red', 'blue'])
        position = random.choice(['left', 'right'])
        size = random.choice(['big', 'small'])
        objects.append({'type': obj_type, 'color': color, 'position': position, 'size': size}) 
    referring_expression = random.choice(['cup', 'bottle'])
    return objects, referring_expression  

if __name__ == "__main__":
    num_scenes = 1000
    scenes = []
    for _ in range(num_scenes):
        scene, ref_expr = generate_scene()
        scenes.append({'scene': scene, 'referring_expression': ref_expr})
    
    with open('generated_scenes.csv', 'w', newline='') as csvfile:
        fieldnames = ['scene', 'referring_expression']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for scene in scenes:
            writer.writerow(scene)

