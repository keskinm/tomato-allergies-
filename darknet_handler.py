import subprocess
import os
import shutil
import argparse


def create_sets_pointer_file():
    sets_pointer_opened_file = open('./cfg/tomato.data', "w")
    content = 'classes= 1\n' \
              'train  = {wd}/data/formated/train.txt\n' \
              'valid  = {wd}/data/formated/test.txt\n' \
              'names = cfg/tomato.names\n' \
              'backup = backup'.format(wd=os.getcwd())
    sets_pointer_opened_file.write(content)
    sets_pointer_opened_file.close()


def main(install):
    if install:
        commands = ['wget https://github.com/AlexeyAB/darknet/archive/master.zip',
                    'unzip master.zip -d .']
        for command in commands:
            subprocess.run(command, check=False, shell=True)
        os.remove('master.zip')
        darknet_dir = './darknet-master'
        subprocess.run('make', check=False, shell=True, cwd=darknet_dir)
        create_sets_pointer_file()
        darknet_cfg_dir = '{}/cfg'.format(darknet_dir)
        shutil.copy('./cfg/tomato.data', darknet_cfg_dir)
        shutil.copy('./cfg/tomato.cfg', darknet_cfg_dir)
        shutil.copy('./cfg/tomato.names', darknet_cfg_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action='store_true', help="prepare data")
    args = parser.parse_args()
    args = vars(args)
    main(**args)








