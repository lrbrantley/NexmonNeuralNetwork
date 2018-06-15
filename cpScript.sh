#! /bin/bash

COUNT=1
while [ $COUNT -le 226 ]; do
   for file in ../magImgCreate/*left$COUNT.png; do
	echo $file
	cp $file .
   done
   let COUNT=COUNT+14
done
