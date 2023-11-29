#!/bin/bash

python eval_qoe.py --logfile Pensieve_0.9mbps_100ms_demo.log --content LOL --quality ultra
python eval_qoe.py --logfile NAS_ultra_0.9mbps_100ms_demo.log --content LOL --quality ultra
python eval_qoe.py --logfile MPC_0.9mbps_100ms_demo.log --content LOL --quality ultra
python eval_qoe.py --logfile BB_0.9mbps_100ms_demo.log --content LOL --quality ultra