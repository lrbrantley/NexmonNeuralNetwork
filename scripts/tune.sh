#!/bin/sh

matrixVar() {
    local flag="$1"
    local values="$2"
    shift 2
    local maxScore=0
    local bestF=0
    for v in $values; do
        out="$($@ $flag "$v" | awk 'NR==1; END{print}')"
        flags="$(printf "$out" | head -n1)"
        score="$(printf "$out" | tail -n1 | cut -d':' -f2)"
        echo "$score > $maxScore" | bc -l > /dev/null &&
            maxScore="$score" && bestF="$flags"
    done
    echo "$bestF"
    echo "Max score is: $maxScore"
}

matrixBatch() {
    #local vs="$(seq 10 10 100)"
    local vs="$(seq 10 10 20)"
    matrixVar -b "$vs" $@
}

matrixConvolution1() {
    #local vs="$(seq 8 8 256)"
    local vs="$(seq 32 8 40)"
    matrixVar -1 "$vs" $@
}

matrixConvolution2() {
    #local vs="$(seq 8 8 256)"
    local vs="$(seq 128 8 136)"
    matrixVar -2 "$vs" $@
}

out=$(matrixBatch \
          matrixConvolution1 \
          matrixConvolution2 $@ -f)
readonly NUM_FLAGS_IGNORED="$(expr $# + 1)"
readonly FLAGS="$(printf "$out" | head -n1 | cut -d' ' -f $NUM_FLAGS_IGNORED-)"
printf "Best flags are: %s\n" "$FLAGS"
printf "$out\n" | tail -n1
