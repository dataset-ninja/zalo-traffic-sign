import os
import shutil
from collections import defaultdict

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    train_path = "/home/alex/DATASETS/TODO/za_traffic_2020/traffic_train/images"
    test_path = "/home/alex/DATASETS/TODO/za_traffic_2020/traffic_public_test/images"
    ann_json_path = (
        "/home/alex/DATASETS/TODO/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json"
    )
    csv_path = "/home/alex/DATASETS/TODO/za_traffic_2020/traffic_train/annotation.csv"

    batch_size = 30

    ds_name_to_data = {"train": train_path, "test": test_path}

    def create_ann(image_path):
        labels = []

        image_name = get_file_name_with_ext(image_path)
        img_height = image_name_to_shape[image_name][0]
        img_wight = image_name_to_shape[image_name][1]

        street_value = image_name_to_shape[image_name][2]
        street = sly.Tag(street_meta, value=street_value)

        ann_data = image_name_to_ann_data[get_file_name_with_ext(image_path)]
        for curr_ann_data in ann_data:
            category_id = int(curr_ann_data[0])
            obj_class = idx_to_class[category_id]
            bbox_coord = curr_ann_data[1]
            rectangle = sly.Rectangle(
                top=int(bbox_coord[1]),
                left=int(bbox_coord[0]),
                bottom=int(bbox_coord[1] + bbox_coord[3]),
                right=int(bbox_coord[0] + bbox_coord[2]),
            )
            label_rectangle = sly.Label(rectangle, obj_class)
            labels.append(label_rectangle)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[street])

    idx_to_class = {
        1: sly.ObjClass("no entry", sly.Rectangle),
        2: sly.ObjClass("no parking/waiting", sly.Rectangle),
        3: sly.ObjClass("no turning", sly.Rectangle),
        4: sly.ObjClass("max speed", sly.Rectangle),
        5: sly.ObjClass("other prohibition signs", sly.Rectangle),
        6: sly.ObjClass("warning", sly.Rectangle),
        7: sly.ObjClass("mandatory", sly.Rectangle),
    }

    street_meta = sly.TagMeta("street id", sly.TagValueType.ANY_NUMBER)

    ann = load_json_file(ann_json_path)

    image_id_to_name = {}
    image_name_to_ann_data = defaultdict(list)
    image_name_to_shape = {}

    for curr_image_info in ann["images"]:
        image_id_to_name[curr_image_info["id"]] = curr_image_info["file_name"]
        image_name_to_shape[curr_image_info["file_name"]] = (
            curr_image_info["height"],
            curr_image_info["width"],
            curr_image_info["street_id"],
        )

    for curr_ann_data in ann["annotations"]:
        image_id = curr_ann_data["image_id"]
        image_name_to_ann_data[image_id_to_name[image_id]].append(
            [curr_ann_data["category_id"], curr_ann_data["bbox"]]
        )

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=list(idx_to_class.values()),
        tag_metas=[street_meta],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, images_path in ds_name_to_data.items():

        dataset = api.dataset.create(
            project.id, get_file_name(ds_name), change_name_if_conflict=True
        )

        images_names = os.listdir(images_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            img_pathes_batch = []
            for image_name in images_names_batch:
                img_pathes_batch.append(os.path.join(images_path, image_name))

            img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            if ds_name != "test":
                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project
