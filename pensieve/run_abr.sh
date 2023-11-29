##!/bin/bash

source activate nas

# Initialize the variables for the options
t=""
n="75" #number of chunk ex): 75 chunks = 5 min (video lenth / 4sec)

# Parse command-line options using getopts
while getopts "t:n:" opt; do
    case "$opt" in
        t)
            t="$OPTARG"
            ;;
        n)
            n="$OPTARG"
            ;;
        \?)
            echo "Usage: $0 -t [type] -n [num]" >&2
            exit 1
            ;;
    esac
done

# Check the value of 't' and execute the corresponding Python script
if [ "$t" == "p" ]; then
        echo "pensieve (video)" # TODO: jh should load pensieve model and set dnn_mode 0
        python rl_server_no_training.py --total_chunk $n \
                --data_dir ./ --model PENSIEVE_43400_linear_71.10_average.ckpt --reward linear --linear_rebuf 4.3 --smooth 1 \
                --bitrate 400 800 1200 2400 4800 --dnn_mode 0 --buffer_threshold 15 \
                --content game --quality ultra1  
elif [ "$t" == "n" ]; then
        echo "nas (video+DNN)"
        python rl_server_no_training.py --total_chunk $n \
                --data_dir ./ --model NAS_40000_linear_104.51_False_ultra1_average.ckpt --reward linear --linear_rebuf 4.3 --smooth 1 \
                --bitrate 400 800 1200 2400 4800 --dnn_mode 1 --buffer_threshold 15 \
                --content game --quality ultra1
elif [ "$t" == "r" ]; then
        echo "replay trace"
        python3 mimic_abr.py ./trace_log/log_norway_train3
elif [ "$t" == "m" ]; then
        echo "robust mpc"
        python mpc_server.py --total_chunk $n \
                --data_dir ./ --reward linear --linear_rebuf 4.3 --smooth 1 \
                --bitrate 400 800 1200 2400 4800 --dnn_mode 1 \
                --content game --quality ultra1
elif [ "$t" == "s" ]; then
        echo "simple server (other baseline abr algos: BB, BOLA, etc)"
        python simple_server.py --total_chunk $n \
                --data_dir ./ \
                --bitrate 400 800 1200 2400 4800 \
                --content game --quality ultra1          
else
    echo "Invalid type specified."
fi