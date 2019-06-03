#!/usr/bin/python

import csv
import sys
from os import path
import numpy as np
import os
import imageio
import argparse
import sys

def parse_args():
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('csv', type=argparse.FileType('r'), help='CSV file')
        parser.add_argument('outformat', nargs='?', help='Output filename format')
        parser.add_argument('--version', action='version', version='%(prog)s 1.1')
        args = parser.parse_args()
        if args.outformat is None:
                name, ext = path.splitext(args.csv.name)
                args.outformat = "%s-%%d.png" % name
        return args

def complex_to_float(c):
        return float(c.replace('(', '').replace('+0j)', '').replace('0j', '0'))

def read_file_floats(inFile):
        out = []
        minV = sys.maxint
        maxV = -minV
        for row in csv.reader(inFile):
                # TODO: Parameterize columns taken
                trimmed = row[3:31]+row[38:65]
                # TODO: Error if column count wrong
                floats = [complex_to_float(s) for s in trimmed]
                out.append(floats)
                minV = min(min(floats), minV)
                maxV = max(max(floats), maxV)
        return (out, minV, maxV)

def scale_row(dat, minV, maxV):
        offset=-minV
        scale=255/(maxV + offset)
        return [int((i + offset)*scale) for i in dat]

def preprocess_file(inFile, outFormat):
        i = 1
        out = []
        data, minV, maxV = read_file_floats(inFile)
        for row in data:
                out.append(scale_row(row, minV, maxV))
                # TODO: Parmaterize
                if len(out) >= 30: #400:
                        imageio.imwrite(outFormat % i,
                                        np.array(out, dtype=np.uint8))
                        i += 1
                        out = []
        if i == 1:
                print("Did not find more than %d rows in file %s." %
                      (len(out), inFile.name))

if __name__ == "__main__":
        args = parse_args()
        preprocess_file(args.csv, args.outformat)
