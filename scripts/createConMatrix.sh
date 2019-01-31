#! /bin/bash

rm ./confusion/*.txt

for file in ./data/validation/amanda/*.png; do
	python3 makeConfMatrix.py weights.best.hdf5 $file >> ./confusion/amanda.txt
done

Amanda=`grep amanda -o ./confusion/amanda.txt | wc -l`
aTotal=`cat ./confusion/amanda.txt | wc -l`
Am_An=`grep andreas -o ./confusion/amanda.txt | wc -l`
Am_Em=`grep empty -o ./confusion/amanda.txt | wc -l`
Am_Lu=`grep lucy -o ./confusion/amanda.txt | wc -l`
Am_Ro=`grep robert -o ./confusion/amanda.txt | wc -l`

for file in ./data/validation/andreas/*.png; do
	python3 makeConfMatrix.py weights.best.hdf5 $file >> ./confusion/andreas.txt
done

An=`grep andreas -o ./confusion/andreas.txt | wc -l`
anTotal=`cat ./confusion/andreas.txt | wc -l`
An_Am=`grep amanda -o ./confusion/andreas.txt | wc -l`
An_Em=`grep empty -o ./confusion/andreas.txt | wc -l`
An_Lu=`grep lucy -o ./confusion/andreas.txt | wc -l`
An_Ro=`grep robert -o ./confusion/andreas.txt | wc -l`

for file in ./data/validation/empty/*.png; do
	python3 makeConfMatrix.py weights.best.hdf5 $file >> ./confusion/empty.txt
done

Empty=`grep empty -o ./confusion/empty.txt | wc -l`
eTotal=`cat ./confusion/empty.txt | wc -l`
E_Am=`grep amanda -o ./confusion/empty.txt | wc -l`
E_An=`grep andreas -o ./confusion/empty.txt | wc -l`
E_Lu=`grep lucy -o ./confusion/empty.txt | wc -l`
E_Ro=`grep robert -o ./confusion/empty.txt | wc -l`


for file in ./data/validation/lucy/*.png; do
	python3 makeConfMatrix.py weights.best.hdf5 $file >> ./confusion/lucy.txt
done

lucy=`grep lucy -o ./confusion/lucy.txt | wc -l`
lTotal=`cat ./confusion/lucy.txt | wc -l`
L_Am=`grep amanda -o ./confusion/lucy.txt | wc -l`
L_An=`grep andreas -o ./confusion/lucy.txt | wc -l`
L_E=`grep empty -o ./confusion/lucy.txt | wc -l`
L_Ro=`grep robert -o ./confusion/lucy.txt | wc -l`

for file in ./data/validation/robert/*.png; do
	python3 makeConfMatrix.py weights.best.hdf5 $file >> ./confusion/robert.txt
done

Robert=`grep robert -o ./confusion/robert.txt | wc -l`
rTotal=`cat ./confusion/robert.txt | wc -l`
R_Am=`grep amanda -o ./confusion/robert.txt | wc -l`
R_An=`grep andreas -o ./confusion/robert.txt | wc -l`
R_Em=`grep empty -o ./confusion/robert.txt | wc -l`
R_Lu=`grep lucy -o ./confusion/robert.txt | wc -l`

echo "amanda = $Amanda, total = $aTotal"
echo "andreas = $Am_An, empty = $Am_Em, lucy = $Am_Lu, robert = $Am_Ro"

echo "andreas = $An, total = $anTotal"
echo "amanda = $An_Am, empty = $An_Em, lucy = $An_Lu, robert = $An_Ro"

echo "empty = $empty, total = $eTotal"
echo "amanda = $E_Am, andreas = $E_An, lucy = $E_Lu, robert = $E_Ro"

echo "lucy = $lucy, total = $lTotal"
echo "amanda = $L_Am, andreas = $L_An, empty = $L_E, robert = $L_Ro"

echo "robert = $Robert, total = $rTotal"
echo "amanda = $R_Am, andreas = $R_An, empty = $R_Em, lucy = $R_Lu"
