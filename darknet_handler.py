import subprocess
import os
import shutil
import argparse
from compute_metrics import compute_metrics


def create_sets_pointer_file():
    sets_pointer_opened_file = open('./cfg/tomato.data', "w")
    content = 'classes= 1\n' \
              'train  = {wd}/data/formated/train.txt\n' \
              'valid  = {wd}/data/formated/test.txt\n' \
              'names = cfg/tomato.names\n' \
              'backup = backup'.format(wd=os.getcwd())
    sets_pointer_opened_file.write(content)
    sets_pointer_opened_file.close()


def main(install, train, test, ckpts_file_path, detection_threshold, gpu):
    ckpts_file_path = ckpts_file_path.replace('./', os.getcwd()+'/')
    darknet_dir = './darknet-master'

    if install:
        commands = ['wget https://github.com/AlexeyAB/darknet/archive/master.zip',
                    'unzip master.zip -d .']
        for command in commands:
            subprocess.run(command, check=False, shell=True)
        os.remove('master.zip')
        subprocess.run('make', check=False, shell=True, cwd=darknet_dir)
        create_sets_pointer_file()
        darknet_cfg_dir = '{}/cfg'.format(darknet_dir)
        shutil.copy('./cfg/tomato.data', darknet_cfg_dir)
        shutil.copy('./cfg/tomato.cfg', darknet_cfg_dir)
        shutil.copy('./cfg/tomato.names', darknet_cfg_dir)
        with open(os.path.join(darknet_dir, 'Makefile'), "r") as makefile_opened_file:
            content = makefile_opened_file.read()
            if gpu:
                content = content.replace('GPU=0', 'GPU=1')
            modified_content = content.replace('OPENCV=0', 'OPENCV=1')
        with open(os.path.join(darknet_dir, 'Makefile'), "w") as makefile_opened_file:
            makefile_opened_file.write(modified_content)

    if train:
        if ckpts_file_path:
            command = './darknet detector train cfg/tomato.data cfg/tomato.cfg {} -map'.format(ckpts_file_path)
        else:
            command = './darknet detector train cfg/tomato.data cfg/tomato.cfg -map'
        subprocess.run(command, check=False, shell=True, cwd=darknet_dir)

    if test:
        if not ckpts_file_path:
            raise ValueError('Cannot test without ckpt file')
        else:
            test_pointer_file_path = '{}/data/formated/test.txt'.format(os.getcwd())
            command = './darknet detector test cfg/tomato.data cfg/tomato.cfg {ckpts_file_path} ' \
                      '-ext_output < {test_set} > preds.txt -thresh {th}'.format(ckpts_file_path=ckpts_file_path,
                                                                    test_set=test_pointer_file_path,
                                                                    th=detection_threshold)
            subprocess.run(command, check=False, shell=True, cwd=darknet_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action='store_true', help="prepare data")
    parser.add_argument("--gpu", action='store_true', help="prepare data")
    parser.add_argument("--train", action='store_true', help="train on train set, compute map on valid set")
    parser.add_argument("--test", action='store_true', help="test on test set")
    parser.add_argument("--detection-threshold", type=float, default='0.15', help="detection threshold to considere "
                                                                                  "there is an object")
    parser.add_argument("--ckpts-file-path", type=str, default='', help="path to ckpts for train/test")

    args = parser.parse_args()
    args = vars(args)
    main(**args)








