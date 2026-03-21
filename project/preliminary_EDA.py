import os
import sys
# sys.path.append("/Users/kathychen/PycharmProjects/KathyVeritasProj")
# sys.path.append("/Users/kathychen/PycharmProjects/KathyVeritasProj/refer")
sys.path.append(os.path.abspath("./refer"))
from skimage import io
from refer import REFER
from pprint import pprint
import matplotlib.pyplot as plt

data_root = "/Users/kathychen/PycharmProjects/KC-Veritas/refer/data"
refer = REFER(data_root, dataset='refcoco', splitBy='google')

# All image IDs used in this dataset
image_ids = refer.getImgIds()
num_images = len(image_ids)

# All reference IDs (each = one referring expression group)
ref_ids = refer.getRefIds()
num_refs = len(ref_ids)

print(f"Total images used: {num_images}")
print(f"Total referring expressions (refs): {num_refs}")

splits = ['train', 'val', 'test']  # standard splits
split_image_counts = {}
split_ref_counts = {}

for split in splits:
    ref_ids_split = refer.getRefIds(split=split)
    img_ids_split = refer.getImgIds(ref_ids=ref_ids_split)

    split_ref_counts[split] = len(ref_ids_split)
    split_image_counts[split] = len(img_ids_split)

print("\nRef counts by split:")
for k, v in split_ref_counts.items():
    print(f"{k}: {v}")

print("\nImage counts by split:")
for k, v in split_image_counts.items():
    print(f"{k}: {v}")

all_splits = set([refer.Refs[ref_id]['split'] for ref_id in refer.Refs])

print("\nSplits actually present in this dataset:")
print(all_splits)

ref_ids = refer.getRefIds(split='test')
first_ref = refer.loadRefs(ref_ids[0])[0]

print("First image ID:", first_ref['image_id'])
print("Expected file name:", first_ref['file_name'])

# Get all test reference IDs
ref_ids = refer.getRefIds(split='test')
print(f"Total test references: {len(ref_ids)}\n")

# Get all image IDs
image_ids = refer.getImgIds()

total_objects = 0

for img_id in image_ids:
    # annotation IDs = objects in that image
    ann_ids = refer.getAnnIds(image_ids=img_id)
    total_objects += len(ann_ids)

avg_objects_per_image = total_objects / len(image_ids)

print(f"Average number of objects per image: {avg_objects_per_image:.2f}")

num_categories = len(refer.Cats)
print(f"Number of categories: {num_categories}")

for cat_id, name in refer.Cats.items():
    print(cat_id, name)

# Show a few examples (e.g., 3 references)
for i, ref_id in enumerate(ref_ids[:5]):
    print(f"=== Example {i+1} ===")
    
    # Load the first reference of this ref_id
    ref = refer.loadRefs(ref_id)[0]
    
    # Print the reference dictionary (referring expression + metadata)
    pprint(ref)
    
    # Print category label
    print(f"Category: {refer.Cats[ref['category_id']]}")
    
    # Print all referring sentences for this object
    sentences = [s['sent'] for s in ref['sentences']]
    print(f"Referring expressions: {sentences}\n")
    
    # Show the image with bounding box
    plt.figure()
    refer.showRef(ref, seg_box='box')
    plt.show()


# start with training set first for all stats
# basic stats
print(len(refer.getRefIds(split='test')))

ref_ids = refer.getRefIds(split='test')
image_ids = [refer.loadRefs(ref_id)[0]['image_id'] for ref_id in ref_ids]
print("Total references:", len(ref_ids))
print("Unique images with references:", len(set(image_ids)))
