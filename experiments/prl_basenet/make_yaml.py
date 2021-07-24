import os
import sys
from shutil import copyfile

sample = sys.argv[1]
prefix = sys.argv[2]
yaml_dir = os.path.dirname(sys.argv[1])

for file in sys.argv[3:]:
    dst_file = prefix + file + '.yaml'
    copyfile(sample, os.path.join(yaml_dir, dst_file))
    print('Copy [%s] created.' % dst_file)