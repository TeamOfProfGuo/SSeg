import os
import sys

def make_yaml():

    config_dir = './config/'
    postfix = '.yaml'

    src = input('Please enter src:')            # xxx_a
    dst = input('Please enter dst:')            # xxx_b
    yaml_num = int(input('Please enter num:'))  # 8
    rpl_str = input('Please enter rpl_str:')
    dst_str = input('Please enter dst_str:')

    for i in range(yaml_num):
        src_path = os.path.join(config_dir, '%s%d%s' % (src, i+1, postfix))
        dst_path = os.path.join(config_dir, '%s%d%s' % (dst, i+1, postfix))
        if os.path.isfile(src_path):
            with open(src_path, 'r') as f:
                info = f.read()
            info = info.replace(rpl_str, dst_str)
            with open(dst_path, 'w') as f:
                f.write(info)
            print('[Info]: Created %s%d%s' % (dst, i+1, postfix))
        # print(sam_path)
        # print(dst_path)
        # print(rpl_str, dst_str)

def make_filename():
    fir_half = input('Please enter first half: ')
    sec_half = input('Please enter second half: ')
    count = int(input('Please enter file num: '))
    for i in range(count+1):
        print('%s%d%s' % (fir_half, i, sec_half))

if __name__ == '__main__':
    if sys.argv[1] == 'yaml':
        make_yaml()
    elif sys.argv[1] == 'filename':
        make_filename()
    else:
        print('Not implemented.')
