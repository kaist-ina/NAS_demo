import numpy as np
import argparse
from read_sr_bitrate_simple import *

M_IN_K = 1000.0
smooth_factor = 1
linear_rebuf = 4.3

bitrates = [400, 800, 1200, 2400, 4800]

num_vid_chunks = 75
num_dnn_chunks = 5

def eval(opt):
    # log_name = "../NAS_ultra_0.9mbps_400ms.log"
    # log_name = "../Pensieve_0.9mbps_400ms.log"

    cnt = 0
    dnn_chunk = 0    
    rebuf = []    
    bitrate = []
    smooth = []
    
    psnr = []
    
    qoe = []
    
    bitrates = [400, 800, 1200, 2400, 4800]
    eff_psnrs = get_baseline_psnr(content=opt.content, dnn_quality=opt.quality)
    
    last_quality = bitrates[1] #default quality
    with open(opt.logfile, 'r') as f:
        for line in f:
            if cnt == num_vid_chunks: 
                break
            
            parse = line.split()
            
            if str(parse[1]) == 'dnndownload':
                dnn_chunk += 1
                eff_bitrates = get_partial_sr_bitrate(dnn_chunk=dnn_chunk, content=opt.content, dnn_quality=opt.quality)
                eff_psnrs = get_partial_sr_psnr(dnn_chunk=dnn_chunk, content=opt.content, dnn_quality=opt.quality)
    
                continue
            
            
            eff_bitrate = float(parse[1])
            q_idx = bitrates.index(int(eff_bitrate))            
            if dnn_chunk > 0:
                eff_bitrate = eff_bitrates[q_idx]
            
            
            print(eff_bitrate, float(parse[3]))
            psnr.append(eff_psnrs[q_idx])            
            bitrate.append(float(eff_bitrate)/M_IN_K)
            rebuf.append(float(parse[3]))
            smooth.append(smooth_factor * np.abs(bitrate[-1] - last_quality)/M_IN_K)
            
            last_quality = bitrate[-1]
            
            calculate_reward=bitrate[-1]- linear_rebuf * rebuf[-1]- smooth_factor * smooth[-1]
        
            qoe.append(calculate_reward)
            cnt += 1
    
    qoe_np = np.asarray(qoe)
    bitrate_np = np.asarray(bitrate)
    psnr_np = np.asarray(psnr)
    
    print(f"QOE(lin): {np.mean(qoe_np):.2f}, Bitrate util: {np.mean(bitrate_np):.2f}, PSNR: {np.mean(psnr_np):.2f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple eval qoe code')
    parser.add_argument('--logfile', required=True, type=str, help='name of logfile')
    parser.add_argument('--content', required=True,  type=str, help='used for selecting content')
    parser.add_argument('--quality', required=True, type=str, choices=('low', 'medium', 'high', 'ultra'), help='used for selecting device')
    
    opt = parser.parse_args()
    
    # log_name = "../NAS_ultra_0.9mbps_400ms.log"
    # log_name = "../Pensieve_0.9mbps_400ms.log"    
    eval(opt)
