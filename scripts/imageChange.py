import imageio
import sys
from os import path
import numpy as np

name, ext = path.splitext(sys.argv[1])
img = imageio.imread(sys.argv[1])
imageio.imwrite(name + "code" + ext, np.hstack((img[:,1:29], img[:,36:64])))
