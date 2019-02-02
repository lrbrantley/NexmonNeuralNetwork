import imageio
import sys
from os import path
import numpy as np

name, ext = path.splitext(sys.argv[1])
img = imageio.imread(sys.argv[1])
n = 0
m = 51
while n < 255:
	imageio.imwrite(name + str(n) + ext, img[n:m,:])
	n = n + 51
	m = m + 51
