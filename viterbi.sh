#!/bin/bash
# Format is: viterbi.sh input_hmm test_file output_file

/opt/python-3.6/bin/python3 viterbi.py $1 $2 > $3

#output_file Format is:
#observ => state seq lgprob
