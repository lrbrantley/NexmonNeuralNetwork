#!/usr/bin/python

import csv
import sys
from os import path
import numpy as np
import os
import imageio

def scaleRow(row):
        trimmed = row[3:31]+row[38:65]
        #print(trimmed)
        floats = [float(s.replace('(', '').replace('+0j)', '').replace('0j', '0')) for s in trimmed]
        minV = -min(floats)
        scale = 255/(max(floats) + minV)
        return [int((i + minV)*scale) for i in floats]

name, ext = path.splitext(sys.argv[1])
i = 1
with open(sys.argv[1], 'r') as inFile:
        #3-31, 38-65
        out = []
        for row in csv.reader(inFile):
                out.append(scaleRow(row))
                if len(out) >= 400:
                        imageio.imwrite("%s-%d.png" % (name, i), np.array(out, dtype=np.uint8))
                        i += 1
                        out = []
