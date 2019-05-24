#! /bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <data dir> <label> [<pngs> ...]" 1>&2
    echo "If no pngs are provided, then they are assumed to be in the current" 1>&2
    echo "directory." 1>&2
    exit 1
fi

readonly DATA_DIR="$1"
readonly LABEL="$2"
shift 2
images="$@"
if [ $# -eq 0 ]; then
    images="./*.png"
fi

readonly TRAIN_DIR="$DATA_DIR/train/$LABEL"
readonly TEST_DIR="$DATA_DIR/validation/$LABEL"
mkdir -p "$TRAIN_DIR" "$TEST_DIR"

for file in $images; do
	val=$RANDOM
	let "val %= 100"
	if test $val -le 70; then
		mv "$file" "$TRAIN_DIR"
	else
		mv "$file" "$TEST_DIR"
	fi
done
