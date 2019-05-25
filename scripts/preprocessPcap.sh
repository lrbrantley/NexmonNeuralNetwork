#!/bin/sh

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data dir> <pcap> [<pcaps> ...]" 1>&2
    exit 1
fi

readonly SCRIPT_DIR="$(dirname $(realpath $0))"

readonly DATA_DIR="$1"
shift 1

readonly CSI_PARSER="$SCRIPT_DIR/csi_parser.py"
readonly CSV_PARSER="$SCRIPT_DIR/preprocessCSV.py"
readonly DATA_SPLITER="$SCRIPT_DIR/dataSplit.sh"

# readonly PSTR="[=======================================================================]"
# progress() {
#     count=$1
#     pd=$(echo "$1*73/$2" | bc)
#     printf "\r%3d.%1d%% %.${pd}s" $(echo "$1*100/$2" | bc) \
#            $(echo "$1*1000/$2 % 10" | bc) $PSTR
# }

processfile() {
    pcap="$1"
    dir="$(dirname $pcap)"
    base="$(basename $pcap .pcap)"
    label=$(echo "$base" | cut -d_ -f1)
    "$CSI_PARSER" "$pcap" "$dir/$base.csv"
    "$CSV_PARSER" "$dir/$base.csv" "$dir/$base-%d.png"
    "$DATA_SPLITER" "$DATA_DIR" "$label" "$dir/$base"-*.png
}

#i=0
for pcap in $@; do
    #progress $i $#
    processfile "$pcap" &
#    i=$(expr $i + 1)
done
for pcap in $@; do
    wait
done
