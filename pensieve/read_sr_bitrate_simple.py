import sys, os, bisect
import numpy as np
# from option import *

RES2BITRATE = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
NODES = [0,1,2,4,6,8]

#TODO: Copy all files into target directory (pensieve/final/dnn_info/)

def analyze_log(content, dnn_quality):
    quality = {}
    for res in [240, 360, 480, 720]:
        quality['psnr-{}-cwdnn'.format(res)] = []
        quality['psnr-{}-bicubic'.format(res)] = []

        quality['ssim-{}-cwdnn'.format(res)] = []
        quality['ssim-{}-bicubic'.format(res)] = []

        quality['bitrate-{}-cwdnn'.format(res)] = []
        quality['bitrate-{}-cgdnn'.format(res)] = []
        quality['bitrate-{}-bicubic'.format(res)] = RES2BITRATE[res]

    for node in NODES:
        # if opt.dataType == 'div2k':
        #     f = open(os.path.join('logs', 'div2k', opt.dataType, content, str(index), opt.model_name, 'result_quality_detail_{}_100.log'.format(node)))
        # else:
        #     f = open(os.path.join('logs', 'video', content, str(index), opt.model_name, 'result_quality_detail_{}_10.log'.format(node)))

        f = open(f"../sr_training/result/{content}/{dnn_quality}/result_quality_detail_{node}_100.log")
        lines = f.readlines()

        #Parse and read quality (PSNR, SSIM, Bitrate)
        if node == 0: #240, 360, 480, 720
            for res in [240, 360, 480, 720]:
                quality['psnr-{}-cwdnn'.format(res)].append([])
                quality['psnr-{}-bicubic'.format(res)].append([])
                quality['ssim-{}-cwdnn'.format(res)].append([])
                quality['ssim-{}-bicubic'.format(res)].append([])

            count_line = 0
            for line in lines:
                #Header
                if count_line == 0:
                    count_line += 1
                    continue
                else:
                    count_line += 1
                    str_list = line.split('\t')
                    #print(str_list)

                    count_tab = 1
                    for res in [240, 360, 480, 720]:
                        quality['psnr-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['psnr-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2
                        quality['ssim-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['ssim-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2
        elif node == 1:
            for res in [720]:
                quality['psnr-{}-cwdnn'.format(res)].append([])
                quality['psnr-{}-bicubic'.format(res)].append([])
                quality['ssim-{}-cwdnn'.format(res)].append([])
                quality['ssim-{}-bicubic'.format(res)].append([])

            count_line = 0
            for line in lines:
                #Header
                if count_line == 0:
                    count_line += 1
                    continue
                else:
                    count_line += 1
                    str_list = line.split('\t')

                    count_tab = 19
                    for res in [720]:
                        quality['psnr-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['psnr-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2
                        quality['ssim-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['ssim-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2

        else:
            for res in [240, 360, 480]:
                quality['psnr-{}-cwdnn'.format(res)].append([])
                quality['psnr-{}-bicubic'.format(res)].append([])
                quality['ssim-{}-cwdnn'.format(res)].append([])
                quality['ssim-{}-bicubic'.format(res)].append([])

            count_line = 0
            for line in lines:
                #Header
                if count_line == 0:
                    count_line += 1
                    continue
                else:
                    count_line += 1
                    str_list = line.split('\t')

                    count_tab = 1
                    for res in [240, 360, 480]:
                        quality['psnr-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['psnr-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2
                        quality['ssim-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['ssim-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2

    #Calculate SSIM-1(SSIM())
    baseline_bitrate = []
    baseline_ssim = []
    for res in [240, 360, 480, 720]:
        baseline_bitrate.append(quality['bitrate-{}-bicubic'.format(res)])
        baseline_ssim.append(np.mean(quality['ssim-{}-bicubic'.format(res)][-1]))
    baseline_bitrate.append(4800)
    baseline_ssim.append(1)

    def make_mapping():
        def _func(ssim):
            idx = bisect.bisect_left(baseline_ssim, ssim)
            if idx == 0:
                return baseline_bitrate[0] + ((baseline_bitrate[1] - baseline_bitrate[0]) /(baseline_ssim[1] - baseline_ssim[0])) * (ssim - baseline_ssim[0])

            elif idx == len(baseline_ssim):
                return baseline_bitrate[idx-2] + ((baseline_bitrate[idx-1] - baseline_bitrate[idx-2]) /(baseline_ssim[idx-1] - baseline_ssim[idx-2])) * (ssim - baseline_ssim[idx-2])

            else:
                return baseline_bitrate[idx-1] + ((baseline_bitrate[idx] - baseline_bitrate[idx-1]) /(baseline_ssim[idx] - baseline_ssim[idx-1])) * (ssim - baseline_ssim[idx-1])
        return _func

    #Calculate bitrate of SR-SSIM
    ssim_to_bitrate = make_mapping()

    for res in [240, 360, 480, 720]:
        for quality_list in quality['ssim-{}-cwdnn'.format(res)]:
            mapped_bitrate = ssim_to_bitrate(np.mean(quality_list))
            quality['bitrate-{}-cwdnn'.format(res)].append(mapped_bitrate)

    return quality


def analyze_log_old(content, index):
    quality = {}
    for res in [240, 360, 480, 720]:
        quality['psnr-{}-cwdnn'.format(res)] = []
        quality['psnr-{}-bicubic'.format(res)] = []

        quality['ssim-{}-cwdnn'.format(res)] = []
        quality['ssim-{}-bicubic'.format(res)] = []

        quality['bitrate-{}-cwdnn'.format(res)] = []
        quality['bitrate-{}-cgdnn'.format(res)] = []
        quality['bitrate-{}-bicubic'.format(res)] = RES2BITRATE[res]

    for node in NODES:
        # if opt.dataType == 'div2k':
        #     f = open(os.path.join('logs', 'div2k', opt.dataType, content, str(index), opt.model_name, 'result_quality_detail_{}_100.log'.format(node)))
        # else:
        #     f = open(os.path.join('logs', 'video', content, str(index), opt.model_name, 'result_quality_detail_{}_10.log'.format(node)))

        f = open(os.path.join("../sr_training/result/", content, 'ultra/result_quality_detail_{}_100.log'.format(node)))
        lines = f.readlines()

        #Parse and read quality (PSNR, SSIM, Bitrate)
        if node == 0: #240, 360, 480, 720
            for res in [240, 360, 480, 720]:
                quality['psnr-{}-cwdnn'.format(res)].append([])
                quality['psnr-{}-bicubic'.format(res)].append([])
                quality['ssim-{}-cwdnn'.format(res)].append([])
                quality['ssim-{}-bicubic'.format(res)].append([])

            count_line = 0
            for line in lines:
                #Header
                if count_line == 0:
                    count_line += 1
                    continue
                else:
                    count_line += 1
                    str_list = line.split('\t')
                    #print(str_list)

                    count_tab = 1
                    for res in [240, 360, 480, 720]:
                        quality['psnr-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['psnr-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2
                        quality['ssim-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['ssim-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2
        elif node == 1:
            for res in [720]:
                quality['psnr-{}-cwdnn'.format(res)].append([])
                quality['psnr-{}-bicubic'.format(res)].append([])
                quality['ssim-{}-cwdnn'.format(res)].append([])
                quality['ssim-{}-bicubic'.format(res)].append([])

            count_line = 0
            for line in lines:
                #Header
                if count_line == 0:
                    count_line += 1
                    continue
                else:
                    count_line += 1
                    str_list = line.split('\t')

                    count_tab = 19
                    for res in [720]:
                        quality['psnr-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['psnr-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2
                        quality['ssim-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['ssim-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2

        else:
            for res in [240, 360, 480]:
                quality['psnr-{}-cwdnn'.format(res)].append([])
                quality['psnr-{}-bicubic'.format(res)].append([])
                quality['ssim-{}-cwdnn'.format(res)].append([])
                quality['ssim-{}-bicubic'.format(res)].append([])

            count_line = 0
            for line in lines:
                #Header
                if count_line == 0:
                    count_line += 1
                    continue
                else:
                    count_line += 1
                    str_list = line.split('\t')

                    count_tab = 1
                    for res in [240, 360, 480]:
                        quality['psnr-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['psnr-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2
                        quality['ssim-{}-cwdnn'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 1
                        quality['ssim-{}-bicubic'.format(res)][-1].append(float(str_list[count_tab]))
                        count_tab += 2

    #Calculate SSIM-1(SSIM())
    baseline_bitrate = []
    baseline_ssim = []
    for res in [240, 360, 480, 720]:
        baseline_bitrate.append(quality['bitrate-{}-bicubic'.format(res)])
        baseline_ssim.append(np.mean(quality['ssim-{}-bicubic'.format(res)][-1]))
    baseline_bitrate.append(4800)
    baseline_ssim.append(1)

    def make_mapping():
        def _func(ssim):
            idx = bisect.bisect_left(baseline_ssim, ssim)
            if idx == 0:
                return baseline_bitrate[0] + ((baseline_bitrate[1] - baseline_bitrate[0]) /(baseline_ssim[1] - baseline_ssim[0])) * (ssim - baseline_ssim[0])

            elif idx == len(baseline_ssim):
                return baseline_bitrate[idx-2] + ((baseline_bitrate[idx-1] - baseline_bitrate[idx-2]) /(baseline_ssim[idx-1] - baseline_ssim[idx-2])) * (ssim - baseline_ssim[idx-2])

            else:
                return baseline_bitrate[idx-1] + ((baseline_bitrate[idx] - baseline_bitrate[idx-1]) /(baseline_ssim[idx] - baseline_ssim[idx-1])) * (ssim - baseline_ssim[idx-1])
        return _func

    #Calculate bitrate of SR-SSIM
    ssim_to_bitrate = make_mapping()

    for res in [240, 360, 480, 720]:
        for quality_list in quality['ssim-{}-cwdnn'.format(res)]:
            mapped_bitrate = ssim_to_bitrate(np.mean(quality_list))
            quality['bitrate-{}-cwdnn'.format(res)].append(mapped_bitrate)

    return quality

def print_effective_bitrates(content, dnn_quality):
    quality_data = analyze_log(content, dnn_quality)
    
    for res in [240, 360, 480, 720]:
        print(f"{res}p-SR:{quality_data['bitrate-{}-cwdnn'.format(res)]}")
        print(f"{res}p-orig:{quality_data['bitrate-{}-bicubic'.format(res)]}")


def print_qualities(content, dnn_quality):
    quality_data = analyze_log(content, dnn_quality)
    
    for res in [240, 360, 480, 720]:
        
        sr_psnrs = np.mean(quality_data['psnr-{}-cwdnn'.format(res)])
        orig_psnrs = np.mean(quality_data['psnr-{}-bicubic'.format(res)])
        
        print(f"{res}p-SR: {sr_psnrs:.2f}")
        print(f"{res}p-orig: {orig_psnrs:.2f}")        
        
        # print(f"{res}p-SR:{quality_data['ssim-{}-cwdnn'.format(res)]}")
        # print(f"{res}p-orig:{quality_data['ssim-{}-bicubic'.format(res)]}")        
        
    

def get_partial_sr_bitrate(dnn_chunk, content, dnn_quality):
    
    logs = analyze_log(content, dnn_quality)
    quality_data = {}
    
    for res in [240, 360, 480, 720]:
        quality_data[f"{res}p"] = logs[f"bitrate-{res}-cwdnn"]
    quality_data['1080p']=[4800]


    quality_list = []
    
    if dnn_chunk == 5:
        quality_list.append(quality_data['240p'][4])
        quality_list.append(quality_data['360p'][4])
        quality_list.append(quality_data['480p'][4])
        quality_list.append(quality_data['720p'][1])
        quality_list.append(quality_data['1080p'][0])
    elif dnn_chunk == 4:
        quality_list.append(quality_data['240p'][3])
        quality_list.append(quality_data['360p'][3])
        quality_list.append(quality_data['480p'][3])
        quality_list.append(quality_data['720p'][1])
        quality_list.append(quality_data['1080p'][0])
    elif dnn_chunk == 3:
        quality_list.append(quality_data['240p'][2])
        quality_list.append(quality_data['360p'][2])
        quality_list.append(quality_data['480p'][2])
        quality_list.append(quality_data['720p'][1])
        quality_list.append(quality_data['1080p'][0])
    elif dnn_chunk == 2: #360,480,720
        quality_list.append(quality_data['240p'][1])
        quality_list.append(quality_data['360p'][1])
        quality_list.append(quality_data['480p'][1])
        quality_list.append(quality_data['720p'][1])
        quality_list.append(quality_data['1080p'][0])
    elif dnn_chunk == 1: #240
        
        quality_list.append(quality_data['240p'][0])
        quality_list.append(quality_data['360p'][0])
        quality_list.append(quality_data['480p'][0])
        quality_list.append(quality_data['720p'][0])
        quality_list.append(quality_data['1080p'][0])

    return quality_list


def get_baseline_psnr(content, dnn_quality):
    logs = analyze_log(content, dnn_quality)
    quality_data = {}
    
    for res in [240, 360, 480, 720]:
        quality_data[f"{res}p"] = logs[f"psnr-{res}-bicubic"]
    quality_data['1080p']=[100]

    quality_list = []
    
    # if dnn_chunk == 5:
    #     quality_list.append(np.mean(quality_data['240p'][4]))
    #     quality_list.append(np.mean(quality_data['360p'][4]))
    #     quality_list.append(np.mean(quality_data['480p'][4]))
    #     quality_list.append(np.mean(quality_data['720p'][1]))
    #     quality_list.append(np.mean(quality_data['1080p'][0]))
    # elif dnn_chunk == 4:
    #     quality_list.append(np.mean(quality_data['240p'][3]))
    #     quality_list.append(np.mean(quality_data['360p'][3]))
    #     quality_list.append(np.mean(quality_data['480p'][3]))
    #     quality_list.append(np.mean(quality_data['720p'][1]))
    #     quality_list.append(np.mean(quality_data['1080p'][0]))
    # elif dnn_chunk == 3:
    #     quality_list.append(np.mean(quality_data['240p'][2]))
    #     quality_list.append(np.mean(quality_data['360p'][2]))
    #     quality_list.append(np.mean(quality_data['480p'][2]))
    #     quality_list.append(np.mean(quality_data['720p'][1]))
    #     quality_list.append(np.mean(quality_data['1080p'][0]))
    # elif dnn_chunk == 2: #360,480,720
    #     quality_list.append(np.mean(quality_data['240p'][1]))
    #     quality_list.append(np.mean(quality_data['360p'][1]))
    #     quality_list.append(np.mean(quality_data['480p'][1]))
    #     quality_list.append(np.mean(quality_data['720p'][1]))
    #     quality_list.append(np.mean(quality_data['1080p'][0]))
    # elif dnn_chunk == 1: #240
        
    quality_list.append(np.mean(quality_data['240p'][0]))
    quality_list.append(np.mean(quality_data['360p'][0]))
    quality_list.append(np.mean(quality_data['480p'][0]))
    quality_list.append(np.mean(quality_data['720p'][0]))
    quality_list.append(np.mean(quality_data['1080p'][0]))

    return quality_list
        

def get_partial_sr_psnr(dnn_chunk, content, dnn_quality):
    
    logs = analyze_log(content, dnn_quality)
    quality_data = {}
    
    for res in [240, 360, 480, 720]:
        quality_data[f"{res}p"] = logs[f"psnr-{res}-cwdnn"]
    quality_data['1080p']=[100]


    quality_list = []
    
    if dnn_chunk == 5:
        quality_list.append(np.mean(quality_data['240p'][4]))
        quality_list.append(np.mean(quality_data['360p'][4]))
        quality_list.append(np.mean(quality_data['480p'][4]))
        quality_list.append(np.mean(quality_data['720p'][1]))
        quality_list.append(np.mean(quality_data['1080p'][0]))
    elif dnn_chunk == 4:
        quality_list.append(np.mean(quality_data['240p'][3]))
        quality_list.append(np.mean(quality_data['360p'][3]))
        quality_list.append(np.mean(quality_data['480p'][3]))
        quality_list.append(np.mean(quality_data['720p'][1]))
        quality_list.append(np.mean(quality_data['1080p'][0]))
    elif dnn_chunk == 3:
        quality_list.append(np.mean(quality_data['240p'][2]))
        quality_list.append(np.mean(quality_data['360p'][2]))
        quality_list.append(np.mean(quality_data['480p'][2]))
        quality_list.append(np.mean(quality_data['720p'][1]))
        quality_list.append(np.mean(quality_data['1080p'][0]))
    elif dnn_chunk == 2: #360,480,720
        quality_list.append(np.mean(quality_data['240p'][1]))
        quality_list.append(np.mean(quality_data['360p'][1]))
        quality_list.append(np.mean(quality_data['480p'][1]))
        quality_list.append(np.mean(quality_data['720p'][1]))
        quality_list.append(np.mean(quality_data['1080p'][0]))
    elif dnn_chunk == 1: #240
        
        quality_list.append(np.mean(quality_data['240p'][0]))
        quality_list.append(np.mean(quality_data['360p'][0]))
        quality_list.append(np.mean(quality_data['480p'][0]))
        quality_list.append(np.mean(quality_data['720p'][0]))
        quality_list.append(np.mean(quality_data['1080p'][0]))

    return quality_list
    

def get_partial_sr_quality_old(dnn_chunk, content, quality):
    #quality_data = get_sr_quality(content, quality)
    quality_data=get_sr_quality(content,quality)
    #ultra1 is ultra 180912 DNN update
    quality_list = []
    if quality == 'ultra0':
        assert 0 #no more ultra0 deprecate
        assert dnn_chunk >=0 and dnn_chunk <=10
        
        #1080p
    #elif quality == 'ultra1':
    else:#Ultra high medium low
        assert dnn_chunk >= 0 and dnn_chunk <=5
        if dnn_chunk == 5:
            quality_list.append(quality_data['240p'][4])
            quality_list.append(quality_data['360p'][4])
            quality_list.append(quality_data['480p'][4])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 4:
            quality_list.append(quality_data['240p'][3])
            quality_list.append(quality_data['360p'][3])
            quality_list.append(quality_data['480p'][3])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 3:
            quality_list.append(quality_data['240p'][2])
            quality_list.append(quality_data['360p'][2])
            quality_list.append(quality_data['480p'][2])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 2: #360,480,720
            quality_list.append(quality_data['240p'][1])
            quality_list.append(quality_data['360p'][1])
            quality_list.append(quality_data['480p'][1])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 1: #240
            
            quality_list.append(quality_data['240p'][0])
            quality_list.append(quality_data['360p'][0])
            quality_list.append(quality_data['480p'][0])
            quality_list.append(quality_data['720p'][0])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 0:
            quality_list = option.opt.bitrate
    '''
    elif quality == 'high':
        assert dnn_chunk >= 0 and dnn_chunk <=5
        if dnn_chunk == 5:
            quality_list.append(quality_data['240p'][4])
            quality_list.append(quality_data['360p'][4])
            quality_list.append(quality_data['480p'][4])
            quality_list.append(quality_data['720p'][2])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 4:
            quality_list.append(quality_data['240p'][3])
            quality_list.append(quality_data['360p'][3])
            quality_list.append(quality_data['480p'][3])
            quality_list.append(quality_data['720p'][2])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 3:
            quality_list.append(quality_data['240p'][2])
            quality_list.append(quality_data['360p'][2])
            quality_list.append(quality_data['480p'][2])
            quality_list.append(quality_data['720p'][2])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 2: #360,480,720
            quality_list.append(quality_data['240p'][1])
            quality_list.append(quality_data['360p'][1])
            quality_list.append(quality_data['480p'][1])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 1: #240
            quality_list.append(quality_data['240p'][0])
            quality_list.append(quality_data['360p'][0])
            quality_list.append(quality_data['480p'][0])
            quality_list.append(quality_data['720p'][0])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 0:
            quality_list = option.opt.bitrate


    
    elif quality == 'medium':
        assert dnn_chunk >= 0 and dnn_chunk <=3
        if dnn_chunk == 3:
            quality_list.append(quality_data['240p'][4])
            quality_list.append(quality_data['360p'][4])
            quality_list.append(quality_data['480p'][4])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 2: #360,480,720
            quality_list.append(quality_data['240p'][2])
            quality_list.append(quality_data['360p'][2])
            quality_list.append(quality_data['480p'][2])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 1: #240
            quality_list.append(quality_data['240p'][0])
            quality_list.append(quality_data['360p'][0])
            quality_list.append(quality_data['480p'][0])
            quality_list.append(quality_data['720p'][0])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 0:
            quality_list = option.opt.bitrate
    elif quality == 'low':
        assert dnn_chunk >= 0 and dnn_chunk <=1
        if dnn_chunk == 1: #240
            quality_list.append(quality_data['240p'][4])
            quality_list.append(quality_data['360p'][4])
            quality_list.append(quality_data['480p'][4])
            quality_list.append(quality_data['720p'][1])
            quality_list.append(quality_data['1080p'][0])
        elif dnn_chunk == 0:
            quality_list = option.opt.bitrate
    
    '''

    #IF SR is smaller don't do it
    for i in range(len(quality_list)):
        if option.opt.bitrate[i]<quality_list[i]:
            pass
        else:
            quality_list[i]=option.opt.bitrate[i]

    return quality_list


def print_bitrate():
    # content_list = ['beauty', 'comedy', 'cook', 'entertainment', 'game', 'music', 'news', 'sport', 'technology']
    # content_idx_list = [1,2,3]
    content_list = ['LOL']
    content_idx_list = [1]


    for content in content_list:
        print(content)
        for res in [240, 360, 480, 720]:
        # for res in [720]:
            print(res)
            qualities = []
            for content_idx in content_idx_list:
                quality = analyze_log(content, content_idx)
                print(quality['bitrate-{}-cwdnn'.format(res)])
                print(quality['bitrate-{}-cwdnn'.format(res)][-1])
                qualities.append(quality['bitrate-{}-cwdnn'.format(res)][-1])
                #qualities += np.array(quality['bitrate-{}-cwdnn'.format(res)])
                #qualities += np.array(quality['ssim-{}-cwdnn'.format(res)])
                #print(np.mean(quality['ssim-{}-cwdnn'.format(res)][-1]))
                #print(np.mean(quality['ssim-{}-cwdnn'.format(res)][-1]))
                #list_.append(quality['bitrate-{}-cwdnn'.format(res)][-1])
            #print(qualities/len(content_idx_list))
            #qualities = qualities/len(content_idx_list)
    print(np.mean(qualities))

def print_ssim():
    #content_list = ['beauty', 'comedy', 'cook', 'entertainment', 'game', 'music', 'news', 'sport', 'technology']
    #content_idx_list = [1,2,3]
    content_list = ['cook']
    content_idx_list = [2]

    ssim_list_ = {}
    ssim_list_['sr'] = {}
    ssim_list_['non_sr'] = {}
    for res in [240, 360, 480, 720]:
        ssim_list_['sr'][res] = 0
        ssim_list_['non_sr'][res] = 0

    for content in content_list:
        print(content)
        for res in [240, 360, 480, 720]:
            print(res)
            qualities = 0
            for content_idx in content_idx_list:
                quality = analyze_log(content, content_idx)
                ssim_list_['sr'][res] += np.mean(quality['ssim-{}-cwdnn'.format(res)][-1])
                ssim_list_['non_sr'][res] += np.mean(quality['ssim-{}-bicubic'.format(res)][-1])

    for res in [240, 360, 480, 720]:
        ssim_list_['sr'][res] = ssim_list_['sr'][res]/(len(content_list) * len(content_idx_list))
        ssim_list_['non_sr'][res] = ssim_list_['non_sr'][res]/(len(content_list) * len(content_idx_list))
    print(ssim_list_)

if __name__ == "__main__":
    print_effective_bitrates(content="LOL", dnn_quality="ultra")
    # print_qualities(content="LOL", dnn_quality="ultra") 
    
    for dnn_chunk in [1,2,3,4,5]:
        
        print(f"for dnn chunk idx {dnn_chunk}")
        
        eff_bitrates = get_partial_sr_bitrate(dnn_chunk=dnn_chunk, content="LOL", dnn_quality="ultra")
        eff_psnrs = get_partial_sr_psnr(dnn_chunk=dnn_chunk, content="LOL", dnn_quality="ultra")
        baseline_psnrs = get_baseline_psnr(content="LOL", dnn_quality="ultra")
        
        print(eff_bitrates)
        print(eff_psnrs)
        print(baseline_psnrs)
        
    # print_bitrate()
    # print_ssim()
