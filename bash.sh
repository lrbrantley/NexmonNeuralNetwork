#! /bin/bash

for file in ./*.png; do
	val=$RANDOM
	let "val %= 100"
	if test $val -le 70; then
		mv "$file" ./data/train
	else
		mv "$file" ./data/validation
	fi
done
