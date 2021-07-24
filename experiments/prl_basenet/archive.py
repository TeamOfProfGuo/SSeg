import os
import sys
from shutil import move

def archive_log(dst_dir = './results/', log_dir = './', yaml_dir = './config/'):
    file_list = os.listdir(log_dir)
    # print(file_list)
    for file in file_list:
        if file.endswith('.log'):
            exp = file.split('.')[0]
            dst_path = os.path.join(dst_dir, exp)
            log_path = os.path.join(log_dir, exp + '.log')
            yaml_path = os.path.join(yaml_dir, exp + '.yaml')
            # print(dst_path)
            # print(log_path)
            # print(yaml_path)
            if os.path.isfile(log_path) and os.path.isfile(yaml_path) and os.path.isdir(dst_path):
                move(log_path, dst_path)
                move(yaml_path, dst_path)
                print('Exp [%s] archived.' % exp)

def archive_path():
    assert len(sys.argv) > 2
    key = sys.argv[2]
    dst_dir = os.path.join('./results/', key)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    root = os.path.dirname(dst_dir)
    # print(root, os.listdir(root))
    for dir in os.listdir(root):
        if (len(dir) > len(key)) and (dir.find(key) != -1):
            move(os.path.join(root, dir), os.path.join(root, key))
            print('[%s] moved to [%s].' % (dir, key))


if __name__ == '__main__':
    # print(sys.argv)
    if sys.argv[1] == 'log':
        archive_log()
    elif sys.argv[1] == 'path':
        archive_path()
        

        

