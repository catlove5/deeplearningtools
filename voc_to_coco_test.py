import os
import json
import xml.etree.ElementTree as ET
import argparse
import shutil
'''
VOC2012
    |--Annotations
    |--ImageSets
        |--Main
    |--JPEGImages    
coco
    |--annotations
    |--train2017
    |--test2017
'''

def transform_voc2coco(args):
    voc_root = args.voc_root

    # 获取工程的根绝对路径
    project_root = os.path.realpath(os.curdir)
    while True:
        if ".idea" in os.listdir(project_root):
            break
        else:
            project_root = os.path.join(project_root, "..")

    # 构建COCO完整目录
    coco_root = os.path.join(project_root, "data/coco")
    coco_train = os.path.join(coco_root, "train2017")
    coco_val = os.path.join(coco_root, "val2017")
    coco_test = os.path.join(coco_root, "test2017")
    coco_anno = os.path.join(coco_root, "annotations")
    coco_train_anno = os.path.join(coco_anno, "instances_train2017.json")
    coco_val_anno = os.path.join(coco_anno, "instances_val2017.json")
    coco_test_anno = os.path.join(coco_anno, "instances_test2017.json")
    if not os.path.exists(coco_root):
        os.makedirs(coco_root)
        os.mkdir(coco_train)
        os.mkdir(coco_val)
        os.mkdir(coco_test)
        os.mkdir(coco_anno)

    coco_train_num = len(os.listdir(coco_train))
    coco_val_num = len(os.listdir(coco_val))
    coco_test_num = len(os.listdir(coco_val))
    print(f"train_2017 number: {coco_train_num}")
    print(f"val_2017 number: {coco_val_num}")
    print(f"test_2017 number: {coco_test_num}")

    # voc数据集目录
    voc_anno_dir = os.path.join(voc_root, "Annotations")
    voc_images_dir = os.path.join(voc_root, "JPEGImages")
    voc_train_txt = os.path.join(voc_root, "ImageSets/Main/train.txt")
    voc_val_txt = os.path.join(voc_root, "ImageSets/Main/val.txt")
    voc_test_txt = os.path.join(voc_root, "ImageSets/Main/test.txt")
    overwrite_images = True
    # 复制voc图片到coco
    if overwrite_images:
        # 复制训练集图片
        with open(voc_train_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_name = line.strip() + ".jpg"
                image_path = os.path.join(voc_images_dir, image_name)
                shutil.copy(image_path, os.path.join(coco_train, image_name))

        # 复制验证集图片
        with open(voc_val_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_name = line.strip() + ".jpg"
                image_path = os.path.join(voc_images_dir, image_name)
                shutil.copy(image_path, os.path.join(coco_val, image_name))
        with open(voc_test_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_name = line.strip() + ".jpg"
                image_path = os.path.join(voc_images_dir, image_name)
                shutil.copy(image_path, os.path.join(coco_val, image_name))

    def _extract_anno(fp, mode: str = "train"):
        if mode == "train":
            txt_file = voc_train_txt
        elif mode == "val":
            txt_file = voc_val_txt
        else:
            txt_file = voc_test_txt
        # 预定义VOC检测的20个类别以及超类
        supercategorys = ["vehicles", "household", "animals", "person"]
        vehicles = ["car", "bus", "bicycle", "motorbike", "aeroplane", "boat", "train"]
        household = ["chair", "sofa", "diningtable", "tvmonitor", "bottle", "pottedplant"]
        animals = ["cat", "dog", "cow", "horse", "sheep", "bird"]
        person = ["person"]
        classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        # 预构建coco格式的json文件
        json_file = {"info": [], "license": [], "images": [], "annotations": [],
                     "categories": [{"id": i, "name": class_,
                                     "supercategory": ""} for i, class_ in enumerate(classes)]}
        for i, class_ in enumerate(classes):
            if class_ in vehicles:
                json_file["categories"][i]["supercategory"] = supercategorys[0]
            elif class_ in household:
                json_file["categories"][i]["supercategory"] = supercategorys[1]
            elif class_ in animals:
                json_file["categories"][i]["supercategory"] = supercategorys[2]
            elif class_ in person:
                json_file["categories"][i]["supercategory"] = supercategorys[3]
            else:
                raise "unsupported class"

                # 写入json文件
        with open(txt_file, "r") as f_:
            lines = f_.readlines()
            for line in lines:
                image_xml = line.strip() + ".xml"
                image_xml_path = os.path.join(voc_anno_dir, image_xml)
                xml_obj = ET.parse(image_xml_path)
                root = xml_obj.getroot()
                img_dir = {"file_name": "", "width": 0, "height": 0, "objects": []}
                i = 0
                for eles in root:
                    if eles.tag == "filename":
                        img_dir["file_name"] = eles.text
                    elif eles.tag == "size":
                        for ele in eles:
                            img_dir["width"] = int(ele.text) if ele.tag == "width" else int(img_dir["width"])
                            img_dir["height"] = int(ele.text) if ele.tag == "height" else int(img_dir["height"])
                    elif eles.tag == "object":
                        obj_dir = {"name": "", "bndbox": [], "image_id": int(img_dir["file_name"].split(".")[0]),
                                   "id": i}
                        i = i + 1
                        for ele in eles:
                            obj_dir["name"] = ele.text if ele.tag == "name" else obj_dir["name"]
                            if ele.tag == "bndbox":
                                for pos in ele:
                                    if pos.tag == "xmin":
                                        xmin = int(pos.text)
                                    elif pos.tag == "xmax":
                                        xmax = int(pos.text)
                                    elif pos.tag == "ymin":
                                        ymin = int(pos.text)
                                    elif pos.tag == "ymax":
                                        ymax = int(pos.text)
                                    else:
                                        raise "unsupported pose"
                                obj_dir["bndbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
                        img_dir["objects"].append(obj_dir)
                        json_file["annotations"].append({"id": obj_dir["id"],
                                                         "image_id": obj_dir["image_id"],
                                                         "category_id": classes.index(obj_dir["name"]),
                                                         "segmentation": [],
                                                         "area": float(obj_dir["bndbox"][2] * obj_dir["bndbox"][3]),
                                                         "bbox": obj_dir["bndbox"],
                                                         "iscrowd": 0})
                    else:
                        continue
                json_file["images"].append({"file_name": img_dir["file_name"],
                                            "height": img_dir["height"],
                                            "width": img_dir["width"],
                                            "id": int(img_dir["file_name"].split(".")[0])})
            json.dump(json_file, fp)

    # 生成coco的annotation标注文件
    override_anno = True
    # 有一个文件不存在或者需要覆盖掉之前的标注
    if not os.path.exists(coco_train_anno) or not os.path.exists(coco_val_anno) or not os.path.exists(coco_test_anno):
        with open(coco_train_anno, "w") as f:
            _extract_anno(f, "train")

        with open(coco_val_anno, "w") as f:
            _extract_anno(f, "val")

        with open(coco_test_anno, "w") as f:
            _extract_anno(f, "val")

    print("-" * 30 + "finish" + "-" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_root', type=str, default="VOC2012")
    args = parser.parse_args()
    transform_voc2coco(args)

