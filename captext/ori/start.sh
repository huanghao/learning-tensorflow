#!/bin/bash

if [ $# != 1 ] ; then
    echo "USAGE: $0 Target Msg"
    echo " e.g.: $0 修改了训练batch_size"
    exit 1;
else
    if [ -d "log" ]; then
        backup_date=`date '+%m%d_%H%M'`
        mv log log_$backup_date
    fi

    mkdir log && echo "$1" >> log/README && python train.py
fi
