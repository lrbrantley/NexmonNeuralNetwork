#! /bin/bash

readonly DATA_DIR=$1
readonly WEIGHTS=$2

if ! [ $# -eq 2 -a -d $DATA_DIR -a -f $WEIGHTS ]; then
    echo "Usage: $0 <data dir> <weights file>" 1>&2
    exit 1
fi

readonly BASE_DIR="$(dirname $(realpath $0))"
readonly TMP="$(mktemp).txt"
readonly TEST_DIR="$DATA_DIR/validation"
readonly BINS="$(ls $TEST_DIR | sort)"

for b in $BINS; do
    for file in $TEST_DIR/$b/*.png; do
	    python3 $BASE_DIR/makeConfMatrix.py $WEIGHTS $file "$BINS"
    done > $TMP 2> /dev/null # TODO: Fix errors instead of redirecting

    echo "$b = $(grep -oc $b $TMP), total = $(wc -l $TMP | cut -d' ' -f1)"
    for other in $BINS; do
        if [ $other = $b ]; then
            continue;
        fi
        printf "%s = %d, " $other $(grep $other -oc $TMP)
    done
    printf "\b\b  \n"
done

rm "$TMP"
