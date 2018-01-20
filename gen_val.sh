#!/bin/bash

cd data/train
for i in `ls`; do find $i/*.* | shuf | head -n 25; done > ../../validation_list.txt