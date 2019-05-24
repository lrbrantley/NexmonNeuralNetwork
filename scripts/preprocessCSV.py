#!/usr/bin/python

import csv
import sys
from os import path
import numpy as np
import os
import imageio
import argparse

def parse_args():
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('csv', type=argparse.FileType('r'), help='CSV file')
        parser.add_argument('outformat', nargs='?', help='Output filename format')
        parser.add_argument('--version', action='version', version='%(prog)s 1.1')
        args = parser.parse_args()
        if args.outformat is None:
                name, ext = path.splitext(args.csv)
                args.outformat = "%s-%%d.png" % name
        return args

def scale_row(row):
        # TODO: Parameterize columns taken
        trimmed = row[3:31]+row[38:65]
        # TODO: Error if column count wrong
        #print(trimmed)
        floats = [float(s.replace('(', '').replace('+0j)', '').replace('0j', '0'))
                  for s in trimmed]
        minV = -min(floats)
        scale = 255/(max(floats) + minV)
        return [int((i + minV)*scale) for i in floats]

def preprocess_file(inFile, outFormat):
        i = 1
        out = []
        for row in csv.reader(inFile):
                out.append(scale_row(row))
                if len(out) >= 400:
                        imageio.imwrite(outFormat % i,
                                        np.array(out, dtype=np.uint8))
                        i += 1
                        out = []

if __name__ == "__main__":
        args = parse_args()
        preprocess_file(args.csv, args.outformat)
