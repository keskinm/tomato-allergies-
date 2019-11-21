import argparse
import json
import csv
import os
import shutil


class Yolo:
    def __init__(self, create_folders, data_annotations_file_path, labels_mapping_file_path, split, data_dir_path):
        self.create_folders = create_folders
        self.annotations = self.parse_annotations(data_annotations_file_path)
        self.mapping = self.label_mapping(labels_mapping_file_path)
        self.split = split
        self.data_dir_path = data_dir_path

    @staticmethod
    def copytree(src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                if not os.path.exists(d):
                    shutil.copy2(s, d)

    @staticmethod
    def normalize_bbox( bbox, h=600, w=600):
        bbox[0] = bbox[0] + bbox[2] / 2
        bbox[1] = bbox[1] + bbox[3] / 2

        bbox[0] /= w
        bbox[1] /= h
        bbox[2] /= w
        bbox[3] /= h
        return bbox

    def prepare_data(self):
        data_dir_path = self.data_dir_path
        formated_data_dir_path = './data/formated'
        os.makedirs(formated_data_dir_path, exist_ok=True)
        os.makedirs(os.path.join(formated_data_dir_path, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(formated_data_dir_path, 'labels'), exist_ok=True)
        self.copytree(data_dir_path, os.path.join(formated_data_dir_path, 'JPEGImages'))

        len_data = len(self.annotations)
        train_cutoff = int(self.split[0]*len_data)
        val_cutoff = int(train_cutoff + self.split[1]*len_data)

        set = 'train'
        pointer_opened_file = open("{}/{}.txt".format(formated_data_dir_path, set), "w")
        for index, image_filename in list(enumerate(self.annotations.keys()))[0:train_cutoff]:
            pointer_opened_file.write(os.path.join(formated_data_dir_path, 'JPEGImages', image_filename) + '\n')
            self.create_label_and_point_it(formated_data_dir_path, image_filename, set)
        pointer_opened_file.close()

            # if 0 <= index <= train_cutoff:
            #     set = 'train'
            # elif train_cutoff <= index <= val_cutoff:
            #     set = 'val'
            # else:
            #     set = 'test'

    def create_label_and_point_it(self, formated_data_dir_path, image_filename, set):
        metadata = self.annotations[image_filename]
        labels = []
        for triplet in metadata:
            label = self.mapping[triplet["id"]]
            if label:
                bbox = triplet["box"]
                bbox = self.normalize_bbox(bbox)
                labels.append([0] + bbox)
        image_filename_without_ext = os.path.splitext(image_filename)[0]
        label_file_path = os.path.join(formated_data_dir_path, 'labels', '{}.txt'.format(image_filename_without_ext))
        label_opened_file = open("{}".format(label_file_path), "w")
        for label in labels:
            str_label = " ".join(map(str, label))
            label_opened_file.write(str_label + '\n')
        label_opened_file.close()

    def run(self):
        if self.prepare_data:
            self.prepare_data()

    def parse_annotations(self, data_annotations_file_path):
        with open(data_annotations_file_path) as json_file:
            annotations = json.load(json_file)
            return annotations

    def label_mapping(self, label_mapping_file_path):
        def tomate_key(dict):
            if "tomate" in str.lower(dict['labelling_name_fr']) or "tomato" in str.lower(dict["labelling_name_en"]):
                return 1
            return 0

        with open(label_mapping_file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            labels = {row['labelling_id']: tomate_key(row) for row in csv_reader}

        return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-folders", action='store_true', help="create folders to use darknet")
    parser.add_argument("--data-annotations-file_path", type=str, default="./data/img_annotations.json", help="path to data annotations file")
    parser.add_argument("--labels-mapping-file-path", type=str, default='./data/label_mapping.csv', help="label mapping file")
    parser.add_argument('--split', nargs=3, default=[0.7, 0.15, 0.15], help='time range to pull scenes from')
    parser.add_argument('--data-dir-path', type=str, default='./data/assignment_imgs', help='path to the directory containing images')
    args = parser.parse_args()
    args = vars(args)
    yolo = Yolo(**args)
    yolo.run()
