from pathlib import Path
import parse
import os
import sys
import shutil
import pickle
import time
import numpy as np

def get_iter(glob):
    ext = os.path.splitext(glob.parts[-1])[1]
    format_string = 'snapshot_iter_{:0>9}' + '{}'.format(ext)
    parsed = parse.parse(format_string, glob.parts[-1])
    return int(parsed[0])

def get_subdirs(p):
    return [x for x in p.iterdir() if x.is_dir()]