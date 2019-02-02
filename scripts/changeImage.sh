#! /bin/bash

for file in ../data/*/*/*.png; do
	echo $file
	python3 imageMakeSmallerSnapshots.py $file
	rm $file
done
