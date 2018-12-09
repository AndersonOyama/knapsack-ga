#!/bin/bash

DIR='./low-dimensional/'
for file in "$DIR"*
do
  echo -e ${file} >> pd.txt
  python3 knapsacj_do.py -int ${file} >> pd.txt
  echo -e "\n" >> pd.txt
done


