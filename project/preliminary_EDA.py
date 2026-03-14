import sys
# sys.path.append("/Users/kathychen/PycharmProjects/KathyVeritasProj")
# sys.path.append("/Users/kathychen/PycharmProjects/KathyVeritasProj/refer")
from skimage import io
from refer import REFER
from pprint import pprint
import matplotlib.pyplot as plt

data_root = "/Users/kathychen/PycharmProjects/KathyVeritasProj/refer/data"
refer = REFER(data_root, dataset='refcoco',  splitBy='google')

ref_ids = refer.getRefIds(split='test')
first_ref = refer.loadRefs(ref_ids[0])[0]

print("First image ID:", first_ref['image_id'])
print("Expected file name:", first_ref['file_name'])

# Get all test reference IDs
ref_ids = refer.getRefIds(split='test')
print(f"Total test references: {len(ref_ids)}\n")

# Show a few examples (e.g., 3 references)
for i, ref_id in enumerate(ref_ids[:3]):
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
