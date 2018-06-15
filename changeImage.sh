#! /bin/bash

for file in ./*.png; do
	#echo $file
	python3 imageChange.py $file
	rm $file
done
