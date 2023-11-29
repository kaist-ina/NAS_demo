
import argparse
import os
import bisect
from pathlib import Path
from datetime import datetime
#constant
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
K_IN_M = 1000.0
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
CHUNK_TIL_VIDEO_END_CAP = 48.0

DNN_REQUEST_SIZE= 200*1000#200KB
DNN_TO_VID_RATE=0.6
DNN_TOTAL_SIZE=3.4*1000*1000#3.4MB

parser = argparse.ArgumentParser(description='pensieve')
SR_PROCESS_TIME = 4 #(4 second)
#train
parser.add_argument('--epoch', default=60000, type=int)
parser.add_argument('--ent_init', default=3, type=float)
parser.add_argument('--ent_decay', default=0.9997, type=float)
parser.add_argument('--actor_lr', default=1e-4, type=float)
parser.add_argument('--actor_lr_decay', default=1, type=float)
parser.add_argument('--critic_lr', default=1e-3, type=float)
parser.add_argument('--critic_lr_decay', default=1, type=float)
parser.add_argument('--num_agents', default=16, type=int)
parser.add_argument('--rand_range', default=1000, type=int)
parser.add_argument('--seq_len', default=200, type=int, help='train batch')

#config
parser.add_argument('--content', required=True,  type=str, help='used for selecting content')
parser.add_argument('--device', default='', type=str, help='used for selecting device')
parser.add_argument('--quality', required=True, type=str, choices=('low', 'medium', 'high', 'ultra0', 'ultra1'), help='used for selecting device')
parser.add_argument('--total_chunk', required=True, type=int, help='used for generating video chunk size file')
parser.add_argument('--default_quality', default=1, type=int)
parser.add_argument('--bw_scale', default=0.1, type=float, help='used for qoe_to_bandwidth test')
parser.add_argument('--save_intv', default=200, type=int, help='used for saving a checkpoint')
parser.add_argument('--special_name', default='', type=str, help='used for logging in plot_results')
parser.add_argument('--partial', action='store_true')

parser.add_argument('--sr_aware', default=0, type=int)

#deprecated soon-- used in sr test.py

#parser.add_argument('--sr_aware', action='store_true', help='apply both super resolution quality and sr-aware pensieve model') #TODO: how to use it?
parser.add_argument('--sr_quality', action='store_true', help='apply super resolution quality, but not sr-aware pensieve model')
parser.add_argument('--rebuf_scale', default=1, type=float)
parser.add_argument('--smooth_scale', default=1, type=float)

#reinforcement learning
parser.add_argument('--s_info', default=6, type=int, help='bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end')
parser.add_argument('--s_len', default=8, type=int, help='take how many frames in the past')
parser.add_argument('--reward', default='linear', choices=('linear', 'log', 'hd'))
parser.add_argument('--linear_rebuf', default=4.3, type=float)
parser.add_argument('--log_rebuf', default=2.6, type=float)
parser.add_argument('--hd_rebuf', default=8, type=float)
parser.add_argument('--smooth', default = 1, type=float)
parser.add_argument('--resolution', default=[240, 360, 480, 720, 1080], nargs='+', type=int)
parser.add_argument('--bitrate', default=[400, 800, 1200, 2400, 4800], nargs='+', type=float)
#TODO: replace sr_bitrate by calling functions: parser.add_argument('--sr_bitrate', default=[984, 1708, 2263, 2890, 4800], nargs='+', type=float)
parser.add_argument('--hd_reward', default=[1, 2, 3, 12, 15], nargs='+', type=float)

#model
parser.add_argument('--test_epoch', default=None, type=int)
parser.add_argument('--model', default=None, type=str, help='restore an existing model for training')
parser.add_argument('--model_baseline', default=None, type=str, help='added for qoe_to_bandwidth_rl')

# Directory
parser.add_argument('--data_dir', default=os.path.join(str(Path.home()), 'OSDI18-pensieve', 'pensieve'), metavar='DIR')
parser.add_argument('--video_dir', default=os.path.join(str(Path.home()), 'OSDI18-pensieve', 'video', '5min'), metavar='DIR')
parser.add_argument('--trace_path', default='trace', metavar='DIR')
parser.add_argument('--train_trace_path', default='cooked_traces', type=str)
parser.add_argument('--valid_trace_path', default='cooked_test_traces', type=str)
parser.add_argument('--test_trace_path', default='./test_sim_traces/', type=str)#'cooked_test_traces'


parser.add_argument('--model_path', default='checkpoint', type=str)


parser.add_argument('--summary_dir', default='./results', type=str)
parser.add_argument('--train_logfile', default='./results/log', type=str)
parser.add_argument('--test_logfile', default='./test_log/log_sim_rl', type=str)
parser.add_argument('--testgraph_dir', default='./test_graph/', type=str)
parser.add_argument('--testlog_dir', default='./test_log/', type=str)
parser.add_argument('--testresult_path', default='test_result', type=str)

parser.add_argument('--video_size_dir', default='./video_sizes')

parser.add_argument('--sr_trace_dir', default='./sr_traces')
parser.add_argument('--sr_trace_name', default='trace')

# config: DNN-Pensieve
#parser.add_argument('--dnn_total_size', default=DNN_TOTAL_SIZE, type=float,help='total size of dnn, size in byte, ex) 4mb=4,000,000')
parser.add_argument('--dnn_penalty',default=1,type=float)
parser.add_argument('--dnn_accelerate',default=0,type=float)
parser.add_argument('--dnn_mode', default=0, choices=(0, 1, 2, 3), type=int,help='0: video - default quality, 1: video+dnn, 2, video+baseline 3: video - sr quality')#dnn mode adds A_DIM 1, baseline= -1
parser.add_argument('--dnn_method', default=0, choices=(0, 1), type=int,help='0 for byte based 1 for rate based')#-1 for original pensieve  0 for byte based, 1 for rate based

# config: Basline algorithms
parser.add_argument('--buffer_threshold',default=15.4187,type=float,help='threshold for baseline buffer, only int (sec)')#baseline buffer threshold
parser.add_argument('--dnn_vid_rate', default=0.6, type=float,help='baseline rate based method, ex) 0.6, 1, 1.5')
parser.add_argument('--dnn_byte_size',default=DNN_REQUEST_SIZE, type=float,help='baseline Byte based method, size in byte ex) 200kb=200,000')

#plot,eval
parser.add_argument('--log_names', default=['log_sim'], nargs='+', type=str)
parser.add_argument('--realworld',default=1,type=int)
parser.add_argument('--scalable',default=1, choices=(0, 1),type=int,help='Turns on/off scalable dnn effect')
#run experiement
parser.add_argument('--cdn_ip', default='localhost', type=str)
parser.add_argument('--abr_algo', default='RL', type=str)
parser.add_argument('--sleeptime', default=3, type=int)
parser.add_argument('--processid', default=2, type=int)
parser.add_argument('--runtime', default=320, type=int)
parser.add_argument('--run_vid_trace', default='trace0.txt', type=str)
parser.add_argument('--random_seed', default=42, type=int)
parser.add_argument('--clockfreq', default=0, type=int)
#qoe_to_bandwidth
parser.add_argument('--dnn_mode_second', default=0, choices=(0, 1, 2, 3), type=int,help='0: video - default quality, 1: video+dnn, 2, video+baseline 3: video - sr quality')#dnn mode adds A_DIM 1, baseline= -1
parser.add_argument('--dnn_method_second', default=0, choices=(0, 1), type=int,help='0 for byte based 1 for rate based')#-1 for original pensieve  0 for byte based, 1 for rate based
parser.add_argument('--model_second', default=None, type=str, help='restore an existing model for training')

opt = parser.parse_args()
opt.seq_len=opt.total_chunk+200

#directory
opt.model_dir = os.path.join(opt.data_dir, opt.model_path)
opt.trace_dir = os.path.join(opt.data_dir, opt.trace_path)
opt.testresult_dir = os.path.join(opt.data_dir, opt.content, opt.device, opt.quality,  opt.testresult_path)
opt.train_trace = os.path.join(opt.trace_dir, opt.train_trace_path)
opt.valid_trace = os.path.join(opt.trace_dir, opt.valid_trace_path)
opt.test_trace = os.path.join(opt.trace_dir, opt.test_trace_path)
opt.testlog_dir=os.path.join(opt.data_dir,opt.content, opt.device, opt.quality, opt.testlog_dir)
opt.train_logfile=os.path.join(opt.data_dir,opt.content, opt.device, opt.quality, opt.train_logfile)
if opt.dnn_mode ==1 and opt.abr_algo in ' RL ':
    opt.realworld_path=os.path.join(opt.data_dir,opt.content,opt.quality,'realworld_'+opt.reward)
else:
    opt.realworld_path=os.path.join(opt.data_dir,opt.content,'realworld_'+opt.reward)

os.makedirs(opt.model_dir, exist_ok=True)
os.makedirs(opt.trace_dir, exist_ok=True)
os.makedirs(opt.testresult_dir, exist_ok=True)
os.makedirs(opt.testgraph_dir, exist_ok=True)
os.makedirs(opt.testresult_dir, exist_ok=True)
os.makedirs(opt.video_size_dir, exist_ok=True)
os.makedirs(opt.testlog_dir, exist_ok=True)
os.makedirs(opt.summary_dir, exist_ok=True)
os.makedirs(opt.realworld_path,exist_ok=True)


#for realworld
os.makedirs('./post_log',exist_ok=True)
os.makedirs(opt.train_logfile[:-3],exist_ok=True)

#setting entropy
if opt.reward == 'linear':
    opt.ent_init = 2
    opt.ent_decay = 0.9999400871493754450993452625

elif opt.reward == 'log':
    opt.ent_init = 2
    opt.ent_decay = 0.999940087
elif opt.reward == 'hd':
    opt.ent_init = 3
    opt.ent_decay = 0.999931978365
else:
    print('unsupported reward')
    sys.exit()

print(opt.testlog_dir)

#validity
assert not (opt.sr_aware and opt.sr_quality) # choose one
