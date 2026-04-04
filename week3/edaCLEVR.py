import json

def load_scenes(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["scenes"]


def compute_stats(scenes):
    total_objects = sum(len(scene["objects"]) for scene in scenes)
    avg_objects = total_objects / len(scenes)

    attributes = {
        "shape": set(),
        "color": set(),
        "size": set(),
        "texture": set()
    }

    for scene in scenes:
        for obj in scene["objects"]:
            attributes["shape"].add(obj["shape"])
            attributes["color"].add(obj["color"])
            attributes["size"].add(obj["size"])
            attributes["texture"].add(obj["material"])

    return avg_objects, attributes

if __name__ == "__main__":
    scenes = load_scenes("/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/CLEVR_train_scenes.json")
    avg_objects, attributes = compute_stats(scenes)

    print("Average objects per scene:", avg_objects)

    print("\nAttribute counts:")
    for key in attributes:
        print(f"{key}: {len(attributes[key])} -> {attributes[key]}")