import subprocess

import numpy as np


def get_frame_rate(filename):
    con = "ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "+ filename

    proc = subprocess.Popen(con, stdout=subprocess.PIPE, shell=True)
    framerateString = str(proc.stdout.read())[2:-5]
    a = int(framerateString.split('/')[0])
    b = int(framerateString.split('/')[1])
    return int(np.round(np.divide(a, b)))
