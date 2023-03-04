#!/usr/local/bin/python3.4
import sys
import re
import getopt
# from tabulate import tabulate # pip3 inst all tabulate!!
import pandas as pd
import numpy as np
import os
import wandb

def parse_performance_log(inputfile):
    
    if not os.path.exists(inputfile): raise FileNotFoundError(f"Did not find: {inputfile}")

    # https://docs.python.org/2/howto/regex.html
    # \[ : [ is a meta char and needs to be escaped 
    # (.*?) : match everything in a non-greedy way and capture it.
    expr = re.compile('\[(.*?)\]')

    data = dict()
    for line in open(inputfile, 'r'):
        # print '\n\n-----';
        strings = expr.findall(line)
        assert(len(strings)>1)
        key = strings[0];
        val = strings[1];
        vals = val.partition(' ')
        unit = vals[2]

        # brings everything on the same scale (ms)
        num_ms = float(vals[0])
        if unit=='mis':
            num_ms = num_ms/1000.0    

        # append time to dictionary entry
        if key in data:
            data[key].append(num_ms)
        else:
            data[key] = [num_ms]

    #--- to sort the data
    # import operator
    # for key in sorted(data, key=operator.itemgetter(0), reverse=True):
    for key in sorted(data):
        wandb.define_metric(f"Runtime_{key}", summary='Mean', hidden=False)
        for e in data[key]:
            wandb.log({f"Runtime_{key}": e})

    #--- create the table
    values = {
        'name': [],
        'median': [],
        'mean': [],
        'std': [],
        'min': [],
        'max': []
    }
    for key in sorted(data):
        values['name'].append(key)
        values['median'].append(np.median(data[key]))
        values['mean'].append(np.mean(data[key]))
        values['std'].append(np.std(data[key]))
        values['max'].append(np.max(data[key]))
        values['min'].append(np.min(data[key]))


    df = pd.DataFrame(values)
    print(df)
    #wandb.log(df)

if __name__ == "__main__":
    # default parameters
    inputfile='htrack_performance.log'
    sortby=0

    # Parse command line options
    opts, args = getopt.getopt(sys.argv[1:],'hi:s:')
    for opt, arg in opts:
        if opt == '-h':
            print('Examples:')
            print('    parselog.py -i performance.log')
            print('    parselog.py -i performance.log -s 1')
            sys.exit(0)
        elif opt in ('-i', '--ifile'):
           inputfile = arg
        elif opt in ('-s', '--sort'):
           sortby = int(arg)
    parse_performance_log(inputfile)