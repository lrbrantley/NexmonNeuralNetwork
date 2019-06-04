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
        parser.add_argument('-r', '--rows', default=400, type=int,
                            help='number of rows per image')
        parser.add_argument('-c', '--columns', default=56, type=int,
                            help='number of columns per image')
        parser.add_argument('--version', action='version', version='%(prog)s 1.1')
        args = parser.parse_args()
        if args.outformat is None:
                name, ext = path.splitext(args.csv.name)
                args.outformat = "%s-%%d.png" % name
        return args

def complex_to_float(c):
        return float(c.replace('(', '').replace('+0j)', '').replace('0j', '0'))

def read_file(inFile):
        out = []
        for row in csv.reader(inFile):
                out.append(row)
        return out

def strip_headers(data):
        if data[0][0] == '':
                return data[1:]
        else:
                return data

def data_to_floats(data):
        out = []
        for row in data:
                out.append([complex_to_float(s) for s in row])
        return out

def float_equ(a, b, E):
        return E > abs(a - b)

# Remove columns which are constant or just increasing by a fixed value
def strip_columns(data):
        EPSILON = 0.7
        incInd = range(len(data[0]))
        incs = [data[1][i] - data[0][i] for i in incInd]
        ref = data[1]
        for row in data[2:]:
                incInd = [i for i in incInd
                          if float_equ(row[i], ref[i] + incs[i], EPSILON)]
                ref = row
        rmInd = [i - ind for ind, i in enumerate(incInd)]
        for row in data:
                for i in rmInd:
                        del row[i]
                #print(row)
        return data

def find_boundries(data):
        minV = min(data[0])
        maxV = max(data[0])
        for row in data[1:]:
                minV = min(min(row), minV)
                maxV = max(max(row), maxV)
        return (minV, maxV)

def scale_row(dat, minV, maxV):
        offset=-minV
        scale=255/(maxV + offset)
        return [int((i + offset)*scale) for i in dat]

def preprocess_file(inFile, outFormat, rows, cols):
        i = 1
        out = []
        data = strip_columns(data_to_floats(strip_headers(read_file(inFile))))
        if len(data[0]) != cols:
                raise AssertionError("Expecting %d columns of data, found %d." %
                                     (cols, len(data[0])))
        minV, maxV = find_boundries(data)
        for row in data:
                out.append(scale_row(row, minV, maxV))
                if len(out) >= rows:
                        imageio.imwrite(outFormat % i,
                                        np.array(out, dtype=np.uint8))
                        i += 1
                        out = []
        if i == 1:
                raise AssertionError("Did not find more than %d rows in file %s." %
                      (len(out), inFile.name))

if __name__ == "__main__":
        args = parse_args()
        preprocess_file(args.csv, args.outformat, args.rows, args.columns)
