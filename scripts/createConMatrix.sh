#! /bin/bash

if [ "$#" -eq "3" ]; then
    readonly LATEX_MODE="1"
    readonly DATA_DIR=$2
    readonly WEIGHTS=$3
else
    readonly LATEX_MODE=""
    readonly DATA_DIR=$1
    readonly WEIGHTS=$2
fi

if ! [ "$#" -ge "2" -a -d "$DATA_DIR" -a -f "$WEIGHTS" ]; then
    echo "Usage: $0 [-latex] <data dir> <weights file>" 1>&2
    exit 1
fi

readonly H5PY_VERSION="$(pip3 show h5py | grep Version | cut -d' ' -f2)"
readonly H5PY_MIN_VERSION="2.8.0"

vercomp () {
    if [[ "$1" == "$3" ]]; then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($3)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if [ "$2" = "=" ]; then
            if ! ((10#${ver1[i]} "$2" 10#${ver2[i]})); then
                return 1
            fi
        else
            if ((10#${ver1[i]} "$2" 10#${ver2[i]})); then
                return 0
            fi
        fi
    done
    [ "$2" = "=" ] && return 0 || return 1
}

if vercomp "$H5PY_VERSION" "<" "$H5PY_MIN_VERSION"; then
    echo "Please upgrade your h5py to $H5PY_MIN_VERSION." 1>&2
    echo "To fix, run: sudo pip3 install --upgrade h5py" 1>&2
fi

readonly BASE_DIR="$(dirname $(realpath $0))"
readonly TMP="$(mktemp)"
readonly TEST_DIR="$DATA_DIR/validation"
readonly BINS="$(ls $TEST_DIR | sort)"

calc() {
    #echo "printf \"$@\n\" | bc -l" 1>&2
    printf "scale=2;$@\n" | bc -l
}

if [ -n "$LATEX_MODE" ]; then
    DELIM=" &"
    LEND="\\\\\\\\\n"
else
    DELIM=", "
    LEND="\n"
fi

printLine() {
    while read item; do
        printf "%10.10s$DELIM " "$item"
    done
    printf "\b\b\b  $LEND"
}

printf "\n$BINS\n" | printLine
for b in $BINS; do
    for file in $TEST_DIR/$b/*.png; do
	    python3 $BASE_DIR/makeConfMatrix.py $WEIGHTS $file "$BINS"
    done > $TMP
    total="$(wc -l $TMP | cut -d' ' -f1)"
    (echo "$b"
     for other in $BINS; do
         calc "$(grep $other -oc $TMP) / $total"
     done
    ) | printLine    
done
printf "\b\b\b"

rm "$TMP"
