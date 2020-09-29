import json


def replace_post_fix(f_r_path, f_w_path):
    f_r = open(f_r_path, "r")
    results = json.load(f_r)
    for item in results:
        item["image_name"] = item["image_name"].replace("jpg", "png")

    f_w = open(f_w_path, "w")
    json.dump(results, f_w)
    f_r.close()
    f_w.close()


replace_post_fix("./result.json","./submit-resnet-101.json")