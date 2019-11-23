import subprocess
import os
import shutil
import argparse


def main(install, train, test, ckpoint_dir):
    if install:
        os.makedirs('./data', exist_ok=True)
        commands = ['git clone https://github.com/AlexeyAB/darknet.git',
                    'cd darknet',
                    'make']
        for command in commands:
            subprocess.run(command, check=False, shell=True, cwd='./data')

        darknet_cfg_dir = './data/darknet/cfg'
        shutil.copy('./cfg/tomato.data', darknet_cfg_dir)
        shutil.copy('./cfg/tomato.cfg', darknet_cfg_dir)
        shutil.copy('./cfg/tomato.names', darknet_cfg_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action='store_true', help="prepare data")
    args = parser.parse_args()
    args = vars(args)
    main(**args)








