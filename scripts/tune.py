#!/usr/bin/python

import sys
import os
sys.path.append('%s/../NeuralNetworks/' %
                os.path.dirname(os.path.realpath(__file__)))
import wifiKerasNeuralNetwork as NN

def matrixVars(flags, args=[]):
    if len(flags) == 0:
        print(str(args))
        return (NN.run(args), args)
    elif flags[0][1] is None:
        return matrixVars(flags[1:], args + [flags[0][0]])
    else:
        flag, values = flags[0]
        maxScore = 0
        bestFlags = None
        for v in values:
            score, f = matrixVars(flags[1:], args + [flag, str(v)])
            print("Score: %f, Flags: %s" % (score, f))
            if score > maxScore:
                maxScore = score
                bestFlags = f
        return (maxScore, bestFlags)

def drange(x, y, jump):
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)
    
def tune(argv=sys.argv):
    return matrixVars([
        ("-b", range(20, 50, 10))#,
        #("-o", optimizers),
        #("-1", None),
        #("-a", activate_fns),
        #("-l", range(8, 256, 24)),
        #("-k", range(2, 4)),
        #("-p", range(2, 4)),
        #("-d", drange(0.05, 0.2, '0.05')),
        #("-2", None),
        #("-a", activate_fns),
        #("-l", range(8, 256, 24)),
        #("-k", range(2, 4)),
        #("-p", range(2, 4)),
        #("-d", drange(0.05, 0.2, '0.05'))
    ], argv + ["-g"])

if __name__ == "__main__":
    score, flags = tune()
    print("Score: %f, Flags: %s" % (score, str(flags)))
