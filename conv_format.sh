#!/bin/bash

#Format is:  cat sys1 | conv_format.sh > sys1_res

/opt/python-3.6.3/bin/python3 conv_format.py > $1
