import sys, os, bisect
import numpy as np
import re
import time
RES2BITRATE = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
NODES = [0,1,2,4,6,8]

#TODO: Copy all files into target directory (pensieve/final/dnn_info/)

def analyze_log(content, index, dnn_quality):
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
        target_path = os.path.join('../super_resolution', 'logs', 'video', content, str(index), '{}_240360480720'.format(dnn_quality), 'result_quality_detail_{}_10.log'.format(node))
        f = open(target_path)
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

#Print average, mean, (sampled based) std: test
"""
for res in [240, 360, 480, 720]:
    for quality_list in quality['ssim-{}-bicubic'.format(res)]:
        print(np.mean(quality_list))
"""
class cache_sr(object):
    def __init__(self):
        self.bitrate={}
        
        
    def check(self,content,quality):
        if content in self.bitrate:
            if quality in self.bitrate[content]:
                return True
                #return cache_sr[content][quality]
            
        return False

    def get(self,content,quality):
        return self.bitrate[content][quality]

    def add(self, content, quality, bitrate_dict):
        if content in self.bitrate:
            self.bitrate[content][quality]=bitrate_dict
            
        else:
            self.bitrate[content]={}
            self.bitrate[content][quality]=bitrate_dict
        
    
global cache_list
cache_list=cache_sr()
def get_sr_quality(content, dnn_quality):
    if dnn_quality == 'ultra1' or dnn_quality =='ultra0':
        dnn_quality='ultra'
    content_idx_list = [1,2,3]
    quality_dict = {}
    if content=='average':
        content_list = ['beauty', 'comedy', 'cook', 'entertainment', 'game', 'music', 'news', 'sport', 'technology']
    else:
        content_list=[content]
    #for content in content_list:
    #print('[INSIDE GET_SR_QUALITY]: ',content)
    
    ####################################NEED TO FIND FASTER WAY TO GET SR QUALITY####################################

    ###CHECK IF CACHING EXIST
    global cache_list
    if cache_list.check(content,dnn_quality):
        
        return cache_list.get(content,dnn_quality)

    else:
        ###
        for res in [240, 360, 480, 720]:
        #for res in [480]:
        #print(res)
            list_ = []
            qualities = 0
            for content_ in content_list:
                for content_idx in content_idx_list:
                    quality = analyze_log(content_, content_idx, dnn_quality)
                    #print(quality['bitrate-{}-cwdnn'.format(res)])
                    #print(quality['bitrate-{}-cwdnn'.format(res)])
                    qualities += np.array(quality['bitrate-{}-cwdnn'.format(res)])
                    #for quality_list in quality['bitrate-{}-cwdnn'.format(res):
                    #list_.append(quality['bitrate-{}-cwdnn'.format(res)][-1])
            #print(qualities/len(content_idx_list))
            qualities = qualities/len(content_idx_list)/len(content_list)
            quality_dict['{}p'.format(res)] = qualities
            #print(qualities)
        quality_dict['1080p']=[4800]
        
        cache_list.add(content,dnn_quality,quality_dict)
        return quality_dict
def check_write(content,dnn_quality):
    if content=='average':
        content_list = ['beauty', 'comedy', 'cook', 'entertainment', 'game', 'music', 'news', 'sport', 'technology']
    else:
        content_list=[content]
        
    content_idx_list = [1,2,3]
    quality_dict = {}
    
    for res in [240, 360, 480, 720]:
    #for res in [480]:
    #print(res)
        list_ = []
        qualities = 0
        for content in content_list:
            for content_idx in content_idx_list:
                quality = analyze_log(content, content_idx, dnn_quality)
                #print(quality['bitrate-{}-cwdnn'.format(res)])
                #print(quality['bitrate-{}-cwdnn'.format(res)])
                qualities += np.array(quality['bitrate-{}-cwdnn'.format(res)])
                #for quality_list in quality['bitrate-{}-cwdnn'.format(res):
                #list_.append(quality['bitrate-{}-cwdnn'.format(res)][-1])
        #print(qualities/len(content_idx_list))
        qualities = qualities/len(content_idx_list)/len(content_list)
        quality_dict['{}p'.format(res)] = qualities
        #print(qualities)
    quality_dict['1080p']=[4800]
    return quality_dict

    
def get_dnn_chunk_size(quality):
    if quality =='ultra1' or quality =='ultra0':
        quality='ultra'
    base_dir = os.path.join('../super_resolution/model/{}'.format(quality)) # for dummy
    # base_dir = os.path.join(f"../sr_training/checkpoint/{content}/{quality}") # for real trained DNN
    
    size_list = []

    for idx in range(5):
        file_path = os.path.join(base_dir, 'DNN_chunk_{}.pth'.format(idx+1))
        # file_path = os.path.join(base_dir, 'DNN_chunk_{}_half.pth'.format(idx+1))        
        file_size = os.path.getsize(file_path) / 1024 #KByte
        size_list.append(file_size)

    return size_list

if __name__ == "__main__":
    content_list = ['beauty', 'comedy', 'cook', 'entertainment', 'game', 'music', 'news', 'sport', 'technology']
    content_idx_list = [1,2,3]
    #content_list = ['beauty']
    #content_idx_list = [1,2]

    #Test get_sr_quality()
    
    #for content in content_list:
    #    print(content,get_sr_quality(content, 'ultra'),check_write(content,'ultra'))
        
            
    #"""
    #Test get_dnn_chunk_size()
    content='average'
    for quality in ['ultra']:#['low', 'medium', 'high', 'ultra']:
        #print(quality,get_dnn_chunk_size(quality))
        #print('sport',check_write('average',quality))
        start=time.time()
        print('start')
        print(content,quality,content,get_sr_quality(content, quality))
        #print('technology',quality,content,get_sr_quality('technology', quality))
        end=time.time()
        print('end',end-start)
    #"""
