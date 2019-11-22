import argparse
import json
import csv
import os
import shutil
import random
from random import shuffle


class TomatoDatasetTool:
    def __init__(self, prepare_data, data_annotations_file_path, labels_mapping_file_path, split, data_dir_path, downsample, upsample, seed):
        self.prepare_data = prepare_data
        self.annotations = self.parse_annotations(data_annotations_file_path)
        self.mapping = self.label_mapping(labels_mapping_file_path)
        self.split = split
        self.data_dir_path = data_dir_path
        self.downsample = downsample
        self.upsample = upsample
        random.seed(seed)

    def compute_class_numbers(self):
        tomatoes_count = 0
        for _, img_filename in enumerate(self.annotations.keys()):
            metadata = self.annotations[img_filename]

            for triplet in metadata:
                tomato = self.mapping[triplet["id"]]
                if tomato:
                    tomatoes_count += 1
                    break
        return (len(self.annotations) - tomatoes_count), tomatoes_count

    def down_sample_data(self, keep_negative_rate=0.05, keep_positive_rate=1.):
        down_sampled_dataset = {}
        total_negatives, total_positives = self.compute_class_numbers()

        keep_negative_n, keep_positive_n = total_negatives*keep_negative_rate, total_positives*keep_positive_rate
        no_tomatoes_count = 0
        tomatoes_count = 0
        for _, img_fname in enumerate(self.annotations.keys()):
            metadata = self.annotations[img_fname]

            for triplet in metadata:
                tomato = self.mapping[triplet["id"]]
                if tomato:
                    if tomatoes_count < keep_positive_n:
                        down_sampled_dataset.setdefault(img_fname, metadata)
                        tomatoes_count += 1
                    break

            else:
                if no_tomatoes_count < keep_negative_n:
                    down_sampled_dataset.setdefault(img_fname, metadata)
                    no_tomatoes_count += 1
        self.annotations = down_sampled_dataset

    @staticmethod
    def copytree(src, dst, symlinks=False, ignore=None):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                if not os.path.exists(d):
                    shutil.copy2(s, d.replace('jpeg', 'jpg'))

    @staticmethod
    def normalize_bbox( bbox, h=600, w=600):
        bbox[0] = bbox[0] + bbox[2] / 2
        bbox[1] = bbox[1] + bbox[3] / 2

        bbox[0] /= w
        bbox[1] /= h
        bbox[2] /= w
        bbox[3] /= h
        return bbox

    def _prepare_data(self):
        data_dir_path = self.data_dir_path
        formated_data_dir_path = './data/formated'
        os.makedirs(formated_data_dir_path, exist_ok=True)
        os.makedirs(os.path.join(formated_data_dir_path, 'JPEGImages'), exist_ok=True)
        os.makedirs(os.path.join(formated_data_dir_path, 'labels'), exist_ok=True)
        self.copytree(data_dir_path, os.path.join(formated_data_dir_path, 'JPEGImages'))

        len_data = len(self.annotations)
        train_cutoff = round(self.split[0]*len_data)
        val_cutoff = round(train_cutoff + self.split[1]*len_data)

        sets = [('train', 0, train_cutoff), ('val', train_cutoff, val_cutoff), ('test', val_cutoff, len(self.annotations) + 1)]
        data_iterator = list(enumerate(self.annotations.keys()))
        shuffle(data_iterator)

        self.create_gt_files_for_computing_error_rate(data_iterator, formated_data_dir_path, sets)
        self.create_label_files(data_iterator, formated_data_dir_path, sets)

    def create_label_files(self, data_iterator, formated_data_dir_path, sets):
        for set, set_start_idx, set_end_idx in sets:
            labels_pointer_opened_file = open("{}/{}.txt".format(formated_data_dir_path, set), "w")
            for index, image_filename in data_iterator[set_start_idx:set_end_idx]:
                labels_pointer_opened_file.write(os.path.join(os.getcwd(), formated_data_dir_path[2:], 'JPEGImages',
                                                              image_filename.replace('jpeg', 'jpg')) + '\n')
                self.create_label_file(formated_data_dir_path, image_filename)
            labels_pointer_opened_file.close()

    def create_gt_files_for_computing_error_rate(self, data_iterator, formated_data_dir_path, sets):
        for set, set_start_idx, set_end_idx in sets:
            gt_opened_file = open("{}/{}_gt.txt".format(formated_data_dir_path, set), "w")
            for index, image_filename in data_iterator[set_start_idx:set_end_idx]:
                tomatoes = []
                metadata = self.annotations[image_filename]
                for triplet in metadata:
                    tomato = self.mapping[triplet["id"]]
                    if tomato:
                        tomatoes.append(tomato)
                gt_opened_file.write(str(bool(tomatoes)) + '\n')
            gt_opened_file.close()

    def create_label_file(self, formated_data_dir_path, image_filename):
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
            if self.downsample:
                self.down_sample_data()
            self._prepare_data()

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
    parser.add_argument("--prepare-data", action='store_true', help="prepare data")
    parser.add_argument("--downsample", action='store_true', help="downsampling data")
    parser.add_argument("--upsample", action='store_true', help="upsampling data")
    parser.add_argument("--data-annotations-file_path", type=str, default="./data/img_annotations.json", help="path to data annotations file")
    parser.add_argument("--labels-mapping-file-path", type=str, default='./data/label_mapping.csv', help="label mapping file")
    parser.add_argument('--split', nargs=3, default=[0.7, 0.15, 0.15], help='time range to pull scenes from')
    parser.add_argument('--data-dir-path', type=str, default='./data/assignment_imgs', help='path to the directory containing images')
    parser.add_argument('--seed', type=int, default=43, help='random seed')
    args = parser.parse_args()
    args = vars(args)
    tomato_dataset_tool = TomatoDatasetTool(**args)
    tomato_dataset_tool.run()
