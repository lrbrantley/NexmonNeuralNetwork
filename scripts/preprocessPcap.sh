#!/bin/sh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data dir> (<pcap> | <csv>) [(<pcap> | <csv>) ...]" 1>&2
    echo "If given a pcap, will generate CSV's and keras images from CSI." 1>&2
    echo "If given a CSV, will skip straight to image generation." 1>&2
    echo "Each pcap/csv will produce zero to many images and sort them into" 1>&2
    echo "training and validation bins under <data dir>/" 1>&2
    exit 1
fi

readonly SCRIPT_DIR="$(dirname $(realpath $0))"

readonly DATA_DIR="$1"
shift 1

# Note: These processes rely heavily on disk IO.  Then tend to spend a lot of
# time blocked.  This number should be significantly higher than the total number
# of available cores for optimal performance.
# On the other hand, this thing can easily fork-bomb.  Be careful.
readonly PROCESS_CHECK_COUNT=24

# In percentage (Not a hard limit, more of a target)
readonly CPU_LIMIT=85

# Sets scheduling priority (limits impact of fork bombs)
# Higher numbers mean lower priority and less performance
# Lower numbers mean higher priority and more risk of fork bombing
readonly NICENESS="+19"

readonly CSI_PARSER="$SCRIPT_DIR/csi_parser.py"
readonly CSV_PARSER="$SCRIPT_DIR/preprocessCSV.py"
readonly DATA_SPLITER="$SCRIPT_DIR/dataSplit.sh"

readonly PCAP_MINE_TYPE="application/vnd.tcpdump.pcap; charset=binary"
readonly CSV_MINE_TYPE="text/plain; charset=us-ascii"

wait_for_cpu_usage()
{
    current=$(mpstat 1 1 | tail -n 1 | awk '$12 ~ /[0-9.]+/ { print int(100 - $12 + 0.5) }')
    while [ "$current" -ge "$1" ]; do
        current=$(mpstat 1 1 | tail -n 1 | awk '$12 ~ /[0-9.]+/ { print int(100 - $12 + 0.5) }')
        sleep 1
    done
}

processfile() {
    pcap="$1"
    dir="$(dirname $pcap)"
    base=""
    if [ "$(file -ib "$pcap")" = "$PCAP_MINE_TYPE" ]; then
        base="$(basename $pcap .pcap)"
        nice -n "$NICENESS" "$CSI_PARSER" "$pcap" "$dir/$base.csv"
    elif [ "$(file -ib "$pcap")" = "$CSV_MINE_TYPE" ]; then
        base="$(basename $pcap .csv)"
    else
        echo "File \"$pcap\" is not a Pcap nor a CSV.  No action to do." 1>&2
        exit 1
    fi
    label=$(echo "$base" | cut -d_ -f1)
    nice -n "$NICENESS" "$CSV_PARSER" "$dir/$base.csv" "$dir/$base-%d.png" &&
    nice -n "$NICENESS" "$DATA_SPLITER" "$DATA_DIR" "$label" "$dir/$base"-*.png
}

i=0
for pcap in $@; do
    processfile "$pcap" &
    i=$(expr $i + 1)
    if [ $(expr $i % $PROCESS_CHECK_COUNT) = 0 ]; then
        wait_for_cpu_usage $CPU_LIMIT
    fi
done
while [ $i -gt 0 ]; do
    wait
    i=$(expr $i - 1)
done
