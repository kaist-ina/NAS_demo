#python library
import argparse, os, sys, logging, time, random

#3rd-party library
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import imageio
#import cv2
from skimage.metrics import structural_similarity as ssim
#own library
import utility as util

# a. Print average per-layer performance
# b. Return average layer-layer performance
# c. (Optional) Save average per-layer performance graph
# d. (Optional) Save resulted images
def test_quality(opt, dataloader, model, epoch, prefix='', test_channel=3, save_image=False, save_graph=False):
    #default vdsr network
    if opt.model == 'static':
        count = 0
        total_sr_psnr = 0
        total_bicubic_psnr = 0
        total_iter = 0
        nLayer = model.module.nLayer

        for iteration, batch in enumerate(dataloader, 1):
            input, target = Variable(batch[0], volatile=True).float(), Variable(batch[1], volatile=True).float()

            input = input.cuda()
            target = target.cuda()

            output = model(input)

            for i in range(len(batch[0])):
                count += 1
                output_ = np.moveaxis(np.clip(output.data[i][:,:,...].cpu().numpy(), 0., 1.), 0, -1)
                input_ = np.moveaxis(input.data[i][:,:,...].cpu().numpy(),0, -1)
                target_ = np.moveaxis(target.data[i][:,:,...].cpu().numpy(), 0, -1)

                if test_channel == 1 and output_.shape[2] == 3:
                    print("###UNSUPPORTED COLOR SPACE###")
                    sys.exit()
                    """
                    output_ = cv2.cvtColor(output_, cv2.COLOR_RGB2YCrCb)[:,:,0]
                    input_ = cv2.cvtColor(input_, cv2.COLOR_RGB2YCrCb)[:,:,0]
                    target_ = cv2.cvtColor(target_, cv2.COLOR_RGB2YCrCb)[:,:,0]
                    """

                psnr_sr = util.psnr(output_, target_, opt.border, max_value=1.)
                psnr_bicubic = util.psnr(input_, target_, opt.border, max_value=1.)

                total_sr_psnr += psnr_sr
                total_bicubic_psnr += psnr_bicubic
                total_iter += 1

                if save_image:
                    output_ *= 255
                    input_ *= 255
                    imageio.imwrite('{}/{}_{}.png'.format(os.path.join(opt.resultDir, opt.model_name), prefix, count), output_.astype(np.uint8))
                    imageio.imwrite('{}/{}_bicubic.png'.format(os.path.join(opt.resultDir, opt.model_name), prefix, count), input_.astype(np.uint8))

        average_sr_psnr_dict = {}
        average_sr_psnr_dict[nLayer] = total_sr_psnr / total_iter
        average_bicubic_psnr = total_bicubic_psnr / total_iter

        print("Epoch[{}] PSNR (sr): {:.3f} PSNR (bicubic): {:.3f}".format(epoch, average_sr_psnr_dict[nLayer], average_bicubic_psnr))

        return average_sr_psnr_dict, average_bicubic_psnr

    #dynamic vdsr 1st version
    elif opt.model == 'dynamic1' or opt.model == 'dynamic2':
        count = 0
        outputNode = model.module.outputNode

        total_sr_psnr = [0] * len(outputNode)
        total_bicubic_psnr = 0
        total_iter = 0

        for iteration, batch in enumerate(dataloader, 1):
            input, target = Variable(batch[0], volatile=True).float(), Variable(batch[1], requires_grad=False).float()

            input = input.cuda()
            target = target.cuda()

            count_old = count
            for layer in range(len(outputNode)):
                count = count_old
                output = model(input, outputNode[layer])
                total_psnr = 0

                for idx in range(len(batch[0])):
                    count += 1
                    output_ = np.moveaxis(np.clip(output.data[idx][:,:,...].cpu().numpy(), 0., 1.), 0, -1)
                    input_ = np.moveaxis(input.data[idx][:,:,...].cpu().numpy(),0, -1)
                    target_ = np.moveaxis(target.data[idx][:,:,...].cpu().numpy(), 0, -1)

                    if test_channel == 1 and output_.shape[2] == 3:
                        output_ = cv2.cvtColor(output_, cv2.COLOR_RGB2YCrCb)[:,:,0]
                        input_ = cv2.cvtColor(input_, cv2.COLOR_RGB2YCrCb)[:,:,0]
                        target_ = cv2.cvtColor(target_, cv2.COLOR_RGB2YCrCb)[:,:,0]

                    psnr_sr = util.psnr(output_, target_, opt.border, max_value=1.)
                    total_psnr += psnr_sr

                    #measure total iteration time and bicubic performance
                    if layer == 0:
                        total_iter += 1
                        psnr_bicubic = util.psnr(input_, target_, opt.border, max_value=1.)
                        total_bicubic_psnr += psnr_bicubic

                    total_sr_psnr[layer] += total_psnr

                    #save an image
                    if save_image:
                        output_ *= 255
                        input_ *= 255
                        imageio.imwrite('{}/{}_layer{}_{}.jpg'.format(os.path.join(opt.resultDir, opt.model_name), prefix, outputNode[layer], count), output_.astype(np.uint8))
                        imageio.imwrite('{}/{}_layer{}_{}_bicubic.jpg'.format(os.path.join(opt.resultDir, opt.model_name), prefix, outputNode[layer], count), input_.astype(np.uint8))

        #print perforamnce
        average_sr_psnr_dict = {}
        average_bicubic_psnr = total_bicubic_psnr / total_iter

        for i in range(len(outputNode)):
            average_sr_psnr_dict[outputNode[i] + 2] =  total_sr_psnr[i]/total_iter
            print("Validation-Layer{}) Epoch[{}] PSNR (sr): {:.3f} PSNR (bicubic): {:.3f}".format(outputNode[i] + 2, epoch, average_sr_psnr_dict[outputNode[i] + 2], average_bicubic_psnr))

        #save performance graph
        #TODO : save per-layer performance graph
        """
        if save_graph:
            util.save_psnr_per_layer('vdsr', test_sr_psnr, test_bicubic_psnr, opt.resultDir + '/' + opt.model_name, 'test_psnr_layer_epoch%d'%(epoch))

        average_sr_psnr_dict = {}
        average_bicubic_psnr = total_bicubic_psnr/total_iter
        """

        return average_sr_psnr_dict, average_bicubic_psnr

def test_runtime(opt, model, epoch, width, height, isdummy=False):
    batch = torch.FloatTensor(opt.vbatchSize, opt.nChannel, height, width).random_(0,256)
    input= Variable(batch, volatile=True)

    input = input.cuda()

    if opt.model == 'static':
        start_time = time.perf_counter()
        output = model(input)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) / opt.vbatchSize
        print('[Layer: {}] Width: {} Height: {} Inference time: {:.02e}'.format(model.module.nLayer, width, height, elapsed_time))

    elif opt.model == 'dynamic1' or opt.model == 'dynamic2':
        test_sr_runtime = []
        nLayer = model.module.nLayer
        outputNode = model.module.outputNode

        for i in range(len(outputNode)):
            start_time = time.perf_counter()
            output = model(input, outputNode[i])
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) / opt.vbatchSize
            print('[Layer: {}] Width: {} Height: {} Inference time: {:.02e}'.format(outputNode[i] + 2, width, height, elapsed_time))
            test_sr_runtime.append(elapsed_time)

        if not isdummy:
            util.save_runtime_per_layer('vdsr', test_sr_runtime, opt.resultDir + '/' + opt.model_name, 'test_runtime_layer_epoch%d'%(epoch), width, height)

def test_static(opt, dataloader, model, epoch, logger, save_image, use_ssim):
    count = 0
    total_sr_psnr = 0
    total_baseline_psnr = 0
    if use_ssim:
        total_sr_ssim = 0
        total_baseline_ssim = 0
    total_iter = 0

    #{}_: under gpu {}: under cpu
    for iteration, batch in enumerate(dataloader, 1):
        assert len(batch[0]) == 1

        if opt.network == 'vdsr':
            input, target = Variable(batch[0], volatile=True).float(), Variable(batch[1], volatile=True).float()
            target_np = target.data[0].permute(1,2,0).numpy()
            input_np = input.data[0].permute(1,2,0).numpy()
        else:
            input, upscaled, target = Variable(batch[0], volatile=True).float(), Variable(batch[1], volatile=True).float(), Variable(batch[2], volatile=True).float()
            target_np = target.data[0].permute(1,2,0).numpy()
            upscaled_np= upscaled.data[0].permute(1,2,0).numpy()
            input_np = input.data[0].permute(1,2,0).numpy()
            #print(input_np.shape, upscaled_np.shape, target_np.shape)

        input_ = input.cuda()
        output_ = model(input_)
        #print(model.target_scale)
        output_ = output_.data[0].permute(1,2,0)
        output_ = torch.clamp(output_,0,1)
        output_np = output_.cpu().numpy()
        #print(output_np.shape)
       # output_np = output_.data[0][:,:,...].cpu().numpy()
       # output_np = np.clip(output_np, 0., 1.)
       # output_np = np.moveaxis(output_np, 0, -1)

        if opt.network == 'vdsr':
            psnr_sr = util.psnr(output_np, target_np, opt.border, max_value=1.)
            psnr_baseline = util.psnr(input_np, target_np, opt.border, max_value=1.)
        else:
            psnr_sr = util.psnr(output_np, target_np, opt.border, max_value=1.)
            psnr_baseline = util.psnr(upscaled_np, target_np, opt.border, max_value=1.)

        if use_ssim:
            if opt.network == 'vdsr':
                ssim_sr = ssim(output_np, target_np, multichannel=True)
                ssim_baseline = ssim(input_np, target_np, multichannel=True)
            else:
                ssim_sr = ssim(output_np, target_np, multichannel=True)
                ssim_baseline = ssim(upscaled_np, target_np, multichannel=True)
            total_sr_ssim += ssim_sr
            total_baseline_ssim += ssim_baseline

        total_sr_psnr += psnr_sr
        total_baseline_psnr += psnr_baseline
        total_iter += 1

        if save_image:
            output_np *= 255
            input_np *= 255
            imageio.imwrite('{}/{}_{}.png'.format(opt.imageDir, total_iter, opt.network), output_np.astype(np.uint8))
            imageio.imwrite('{}/{}_{}.png'.format(opt.imageDir, total_iter, opt.resize), input_np.astype(np.uint8))

    average_sr_psnr = total_sr_psnr / total_iter
    average_baseline_psnr = total_baseline_psnr / total_iter

    if use_ssim:
        average_sr_ssim = total_sr_ssim / total_iter
        average_baseline_ssim = total_baseline_ssim / total_iter
        logger.info('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(epoch, average_sr_psnr, average_baseline_psnr, average_sr_ssim, average_baseline_ssim))
        logging.info("Epoch[{}] PSNR (sr): {:.3f} PSNR ({}): {:.3f} MS-SSIM (sr): {:3f} MS-SSIM ({}): {:.3f}".format(epoch, average_sr_psnr, opt.resize, average_baseline_psnr, average_sr_ssim, opt.resize, average_baseline_ssim))
    else:
        logger.info('{}\t{:.3f}\t{:.3f}'.format(epoch, average_sr_psnr, average_baseline_psnr))
        logging.info("Epoch[{}] PSNR (sr): {:.3f} PSNR ({}): {:.3f}".format(epoch, average_sr_psnr, opt.resize, average_baseline_psnr))

#TODO: update ssim as test_static
def test_dynamic(opt, dataloader, model, epoch, logger, save_image, use_ssim):
    #print('### UPDATE TO SUPPORT NEW NETWORK - Refer test_static() ###')
    #sys.exit()
    outputNode = model.outputNode
    total_sr_psnr = {}
    total_baseline_psnr = 0
    total_iter = 0

    #{}_: under gpu {}: under cpu
    for iteration, batch in enumerate(dataloader, 1):
        assert len(batch[0]) == 1

        if opt.network == 'vdsr':
            input, target = Variable(batch[0], volatile=True).float(), Variable(batch[1], volatile=True).float()
            target_np = target.data[0].permute(1,2,0).numpy()
            input_np = input.data[0].permute(1,2,0).numpy()
            psnr_baseline = util.psnr(input_np, target_np, opt.border, max_value=1.)
        else:
            input, upscaled, target = Variable(batch[0], volatile=True).float(), Variable(batch[1], volatile=True).float(), Variable(batch[2], volatile=True).float()
            target_np = target.data[0].permute(1,2,0).numpy()
            upscaled_np= upscaled.data[0].permute(1,2,0).numpy()
            input_np = input.data[0].permute(1,2,0).numpy()
            psnr_baseline = util.psnr(upscaled_np, target_np, opt.border, max_value=1.)

        input_ = input.cuda()

        #target_np = target.data[0].numpy() #convert to numpy array
        #target_np = np.moveaxis(target_np, 0, -1)
        #input_np = input.data[0].numpy() #convert to numpy array
        #input_np = np.moveaxis(input_np, 0, -1)

        total_baseline_psnr += psnr_baseline
        total_iter += 1

        for layer in outputNode:
            output_ = model(input_, layer)
            output_np = output_.data[0][:,:,...].cpu().numpy()
            output_np = np.clip(output_np, 0., 1.)
            output_np = np.moveaxis(output_np, 0, -1)

            psnr_sr = util.psnr(output_np, target_np, opt.border, max_value=1.)

            if layer not in total_sr_psnr:
                total_sr_psnr[layer] = psnr_sr
            else:
                total_sr_psnr[layer] += psnr_sr

            #save an image
            if save_image:
                output_np *= 255
                input_np *= 255
                imageio.imwrite('{}/{}_layer{}_vdsr.png'.format(opt.imageDir, layer, total_iter), output_np.astype(np.uint8))
                imageio.imwrite('{}/{}_layer{}_{}.png'.format(opt.imageDir, layer, total_iter, opt.resize), input_np.astype(np.uint8))

    #print perforamnce
    average_sr_psnr_dict = {}
    average_baseline_psnr = total_baseline_psnr / total_iter

    log_str='{}\t'.format(epoch)

    if opt.network == 'vdsr':
        for layer in outputNode:
            average_sr_psnr_dict[layer] =  total_sr_psnr[layer]/total_iter
            log_str+= '{:.3f}\t'.format(average_sr_psnr_dict[layer])
            logging.info("[Layer{}, Epoch[{}] PSNR (sr): {:.3f} PSNR ({}): {:.3f}".format(layer + 2, epoch, average_sr_psnr_dict[layer], opt.resize, average_baseline_psnr))
        logger.info(log_str)
    else:
        for resblock in outputNode:
            average_sr_psnr_dict[resblock] =  total_sr_psnr[resblock]/total_iter
            log_str+= '{:.3f}\t'.format(average_sr_psnr_dict[resblock])
            logging.info("[Resblock{}, Epoch[{}] PSNR (sr): {:.3f} PSNR ({}): {:.3f}".format(resblock, epoch, average_sr_psnr_dict[resblock], opt.resize, average_baseline_psnr))
        logger.info(log_str)

def test_model(opt, dataloader, model, epoch, logger, save_image=False, use_ssim=False):
    if 'static' in opt.model:
        test_static(opt, dataloader, model, epoch, logger, save_image, use_ssim)
    else:
        test_dynamic(opt, dataloader, model, epoch, logger, save_image, use_ssim)

#TODO: 24, (720, 1280) is hard-coded - need to removed
#total_number means total frame form a video - 24 means 1 sec
def test_video(opt, dataloader, model, epoch, logger, cdf_logger, sr_out, baseline_out, total_number=24):
    timer = util.timer()
    total_iter = 0
    sr_frame_list = []
    baseline_frame_list = []
    sr_ssim_list = []
    baseline_ssim_list = []


    if not sr_out.isOpened() or not baseline_out.isOpened():
        logging.warn('Cannot write')

    #{}_: under gpu {}: under cpu
    for iteration, batch in enumerate(dataloader, 1):
        if total_iter ==  opt.frameNum:
            break

        #timer.tic()
        assert len(batch[0]) == 1
        input, target = Variable(batch[0], volatile=True).float(), Variable(batch[1], volatile=True).float()
        input_ = input.cuda()

        #GPU processing
        output_ = model(input_)
        #torch.cuda.synchronize()
        #logging.info('Process Elapsed Time: {:.3f}'.format(timer.toc()))

        #GPU post-processing
        #timer.tic()
        output_ = output_.data[0].permute(1,2,0)
        output_ = output_ * 255
        output_ = torch.clamp(output_,0,255)
        output_ = torch.round(output_)
        #logging.info('GPU-postprocess Elapsed Time: {:.3f}'.format(timer.toc()))

        #CPU-post-processing
        #timer.tic()
        input = input * 255
        input = input.data[0].permute(1,2,0)
        target = target * 255
        target = target.data[0].permute(1,2,0)
        output = output_.cpu().numpy()
        target = target.numpy()
        input = input.numpy()
        output = output.astype(np.uint8)
        target = target.astype(np.uint8)
        input = input.astype(np.uint8)
        #logging.info('CPU-postprocess Elapsed Time: {:.3f}'.format(timer.toc()))

        #Calculate SSIM
        #timer.tic()
        sr_ssim = ssim(output, target, multichannel=True)
        baseline_ssim = ssim(input, target, multichannel=True)
        sr_ssim_list.append(sr_ssim)
        baseline_ssim_list.append(baseline_ssim)
        logger.info('{}\t{:.3f}\t{:.3f}'.format(total_iter, sr_ssim, baseline_ssim))
        #logging.info('SSIM Elapsed Time: {:.3f}'.format(timer.toc()))

        #Add a frame to a video
        #timer.tic()
        output = output[:,:,[2,1,0]]
        input = input [:,:,[2,1,0]]
        sr_out.write(output)
        baseline_out.write(input)
        #logging.info('Video write Elapsed Time: {:.3f}'.format(timer.toc()))

        if total_iter % 24 == 0:
            logging.info('Process: {} frames done'.format(total_iter))
        total_iter += 1

    #Log average, vairance and cdf
    cdf_logger.info('Average\t{:.3f}\t{:.3f}'.format(np.mean(sr_ssim_list), np.mean(baseline_ssim_list)))
    cdf_logger.info('Variance\t{:.3f}\t{:.3f}'.format(np.var(sr_ssim_list), np.var(baseline_ssim_list)))

    sr_ssim_list = sorted(sr_ssim_list)
    baseline_ssim_list = sorted(baseline_ssim_list)
    key_sr = sorted(list(set(sr_ssim_list)))
    key_baseline = sorted(list(set(baseline_ssim_list)))

    baseline_cdf = util.discrete_cdf(baseline_ssim_list)
    sr_cdf = util.discrete_cdf(sr_ssim_list)

    print(len(key_sr))
    print(len(key_baseline))

    for i in range(max(len(key_sr), len(key_baseline))):
        log_str = ''
        if i < len(key_sr):
            log_str += '{}\t{}\t'.format(key_sr[i], sr_cdf(key_sr[i]))
        if i < len(key_baseline):
            log_str += '{}\t{}\t'.format(key_baseline[i], baseline_cdf(key_baseline[i]))
        cdf_logger.info(log_str)


def save_checkpoint_resolution(opt, model, quality):
    model_out_dir = os.path.join('model_resolution', quality)
    os.makedirs(model_out_dir, exist_ok=True)
    state_dict = model.state_dict()

    index_2_res = [240, 360, 480, 720]
    for index in range(4):
        model_out_path = os.path.join(model_out_dir, 'DNN_chunk_{}.pth'.format(index_2_res[index]))
        total_dict = {}
        dict01 = {k:v for (k,v) in state_dict.items() if '{}.head'.format(index) in k}
        dict02 = {k:v for (k,v) in state_dict.items() if '{}.tail'.format(index) in k}
        dict03 = {k:v for (k,v) in state_dict.items() if '{}.upscale'.format(index) in k}
        dict04 = {k:v for (k,v) in state_dict.items() if '{}.body_end.'.format(index) in k}
        total_dict = {**total_dict, **dict01, **dict02, **dict03, **dict04}

        for idx in range(4):
            partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.{}.0'.format(index, 2*idx) in k}
            total_dict = {**total_dict, **partial_dict}

            partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.{}.0'.format(index, 2*idx+1) in k}
            total_dict = {**total_dict, **partial_dict}

        print(len(total_dict.keys()))
        torch.save(total_dict, model_out_path)

#Save a final NAS model
def save_checkpoint_scalable(opt, model, quality):
    model_out_dir = os.path.join('model', quality)
    os.makedirs(model_out_dir, exist_ok=True)
    state_dict = model.state_dict()

    #Baseblock
    total_dict = {}
    model_out_path = os.path.join(model_out_dir, 'DNN_chunk_1.pth')
    for index in range(4):
        dict01 = {k:v for (k,v) in state_dict.items() if '{}.head'.format(index) in k}
        dict02 = {k:v for (k,v) in state_dict.items() if '{}.tail'.format(index) in k}
        dict03 = {k:v for (k,v) in state_dict.items() if '{}.upscale'.format(index) in k}
        dict04 = {k:v for (k,v) in state_dict.items() if '{}.body_end.'.format(index) in k}
        total_dict = {**total_dict, **dict01, **dict02, **dict03, **dict04}

    #print(len(total_dict.keys()))
    torch.save(total_dict, model_out_path)

    for idx in range(4):
        total_dict ={}
        model_out_path = os.path.join(model_out_dir, 'DNN_chunk_{}.pth'.format(idx+2))

        for index in range(4):
            partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.{}.0'.format(index, 2*idx) in k}
            total_dict = {**total_dict, **partial_dict}

        for index in range(4):
            partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.{}.0'.format(index, 2*idx+1) in k}
            total_dict = {**total_dict, **partial_dict}

        #print(len(total_dict.keys()))
        torch.save(total_dict, model_out_path)

#partial DNN
def save_checkpoint_ultra(opt, model):
    #model_out_path = os.path.join(opt.modelDir, 'epoch_{}.pth'.format(epoch))
    state_dict = model.state_dict()

    #head
    total_dict = {}
    model_out_path = os.path.join('DNN_chunk_1.pth')
    dict01 = {k:v for (k,v) in state_dict.items() if '{}.head'.format(0) in k}
    dict02 = {k:v for (k,v) in state_dict.items() if '{}.tail'.format(0) in k}
    dict03 = {k:v for (k,v) in state_dict.items() if '{}.upscale'.format(0) in k}
    dict04 = {k:v for (k,v) in state_dict.items() if '{}.body_end.'.format(0) in k}
    total_dict = {**dict01, **dict02, **dict03, **dict04}

    for index in range(1,4):
        dict01 = {k:v for (k,v) in state_dict.items() if '{}.head'.format(index) in k}
        dict02 = {k:v for (k,v) in state_dict.items() if '{}.tail'.format(index) in k}
        dict03 = {k:v for (k,v) in state_dict.items() if '{}.upscale'.format(index) in k}
        dict04 = {k:v for (k,v) in state_dict.items() if '{}.body_end.'.format(index) in k}
        total_dict = {**total_dict, **dict01, **dict02, **dict03, **dict04}
    torch.save(total_dict, model_out_path)


    total_dict ={}
    model_out_path = os.path.join('DNN_chunk_2.pth')
    total_dict = {k:v for (k,v) in state_dict.items() if '{}.body.0.0'.format(0) in k}

    for index in range(1,4):
        partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.0.0'.format(index) in k}
        total_dict = {**total_dict, **partial_dict}
    torch.save(total_dict, model_out_path)

    total_dict ={}
    model_out_path = os.path.join('DNN_chunk_3.pth')
    total_dict = {k:v for (k,v) in state_dict.items() if '{}.body.1.0'.format(0) in k}

    for index in range(1,4):
        partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.1.0'.format(index) in k}
        total_dict = {**total_dict, **partial_dict}
    torch.save(total_dict, model_out_path)

    total_dict ={}
    model_out_path = os.path.join('DNN_chunk_4.pth')
    total_dict = {k:v for (k,v) in state_dict.items() if '{}.body.2.0'.format(0) in k}

    for index in range(1,4):
        partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.2.0'.format(index) in k}
        total_dict = {**total_dict, **partial_dict}
    torch.save(total_dict, model_out_path)

    total_dict ={}
    model_out_path = os.path.join('DNN_chunk_5.pth')
    total_dict = {k:v for (k,v) in state_dict.items() if '{}.body.3.0'.format(0) in k}

    for index in range(1,4):
        partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.3.0'.format(index) in k}
        total_dict = {**total_dict, **partial_dict}
    torch.save(total_dict, model_out_path)

    logging.info("Checkpoint saved to {}".format(model_out_path))

def save_dnn_chunk(model_path):
    #model_out_path = os.path.join(opt.modelDir, 'epoch_{}.pth'.format(epoch))

    if not os.path.isfile(model_path):
        sys.exit('no model found at {}'.format(model_path))

    checkpoint = torch.load(model_path)
    state_dict = checkpoint["model"].state_dict()

    model_dir = os.path.dirname(os.path.realpath(model_path))
    #model_dir = './'
    #Debug
    #state_dict = model.state_dict()
    #for name, param in state_dict.items():
    #    print(name)
    #for item in state_dict.itemts())
    #save_checkpoint_partial(opt, model, 0)
    #save_checkpoint_ultra(opt, model)
    #sys.exit()

    #head
    total_dict = {}
    model_out_path = os.path.join(model_dir, 'DNN_chunk_1.pth')
    print(model_out_path)
    for index in range(4):
        dict01 = {k:v for (k,v) in state_dict.items() if '{}.head'.format(index) in k}
        dict02 = {k:v for (k,v) in state_dict.items() if '{}.tail'.format(index) in k}
        dict03 = {k:v for (k,v) in state_dict.items() if '{}.upscale'.format(index) in k}
        dict04 = {k:v for (k,v) in state_dict.items() if '{}.body_end.'.format(index) in k}
        total_dict = {**total_dict, **dict01, **dict02, **dict03, **dict04}
    torch.save(total_dict, model_out_path)

    total_dict ={}
    model_out_path = os.path.join(model_dir, 'DNN_chunk_2.pth')
    total_dict = {k:v for (k,v) in state_dict.items() if '{}.body.0.0'.format(0) in k}

    for index in range(4):
        partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.0.0'.format(index) in k}
        total_dict = {**total_dict, **partial_dict}
    torch.save(total_dict, model_out_path)

    total_dict ={}
    model_out_path = os.path.join(model_dir, 'DNN_chunk_3.pth')
    total_dict = {k:v for (k,v) in state_dict.items() if '{}.body.1.0'.format(0) in k}

    for index in range(4):
        partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.1.0'.format(index) in k}
        total_dict = {**total_dict, **partial_dict}
    torch.save(total_dict, model_out_path)

    total_dict ={}
    model_out_path = os.path.join(model_dir, 'DNN_chunk_4.pth')
    total_dict = {k:v for (k,v) in state_dict.items() if '{}.body.2.0'.format(0) in k}

    for index in range(4):
        partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.2.0'.format(index) in k}
        total_dict = {**total_dict, **partial_dict}
    torch.save(total_dict, model_out_path)

    total_dict ={}
    model_out_path = os.path.join(model_dir, 'DNN_chunk_5.pth')
    total_dict = {k:v for (k,v) in state_dict.items() if '{}.body.3.0'.format(0) in k}

    for index in range(4):
        partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.3.0'.format(index) in k}
        total_dict = {**total_dict, **partial_dict}
    torch.save(total_dict, model_out_path)

    logging.info("Checkpoint saved to {}".format(model_out_path))

#partial DNN
def save_checkpoint_partial(opt, model, index):
    #model_out_path = os.path.join(opt.modelDir, 'epoch_{}.pth'.format(epoch))
    state_dict = model.state_dict()

    #head
    model_out_path = os.path.join('base_{}.pth'.format(index))
    dict01 = {k:v for (k,v) in state_dict.items() if '{}.head'.format(index) in k}
    dict02 = {k:v for (k,v) in state_dict.items() if '{}.tail'.format(index) in k}
    dict03 = {k:v for (k,v) in state_dict.items() if '{}.upscale'.format(index) in k}
    dict04 = {k:v for (k,v) in state_dict.items() if '{}.body_end.'.format(index) in k}
    partial_dict = {**dict01, **dict02, **dict03, **dict04}
    torch.save(partial_dict, model_out_path)

    model_out_path = os.path.join('body0_{}.pth'.format(index))
    partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.0.0'.format(index) in k}
    torch.save(partial_dict, model_out_path)

    model_out_path = os.path.join('body1_{}.pth'.format(index))
    partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.1.0'.format(index) in k}
    torch.save(partial_dict, model_out_path)

    model_out_path = os.path.join('body2_{}.pth'.format(index))
    partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.2.0'.format(index) in k}
    torch.save(partial_dict, model_out_path)

    model_out_path = os.path.join('body3_{}.pth'.format(index))
    partial_dict = {k:v for (k,v) in state_dict.items() if '{}.body.3.0'.format(index) in k}
    torch.save(partial_dict, model_out_path)

    logging.info("Checkpoint saved to {}".format(model_out_path))

#save a checkpoint
def save_checkpoint(opt, model, epoch):
    model_out_path = os.path.join(opt.modelDir, 'epoch_{}.pth'.format(epoch))
    if 'static' in opt.model:
        state = {"epoch": epoch ,"model_type": opt.model, "model": model, "nLayer": opt.nLayer, "nFeat": opt.nFeat, "nChannel": opt.nChannel, "network_type": opt.network}
    else:
        state = {"epoch": epoch ,"model_type": opt.model, "model": model, "nLayer": opt.nLayer, "nFeat": opt.nFeat, "nChannel": opt.nChannel, "outputType": opt.outputType, "minOutput": opt.minOutput, "network_type": opt.network}

    torch.save(state, model_out_path)
    logging.info("Checkpoint saved to {}".format(model_out_path))

#save a model components by components
def save_checkpoint_components(model, epoch):
    if not os.path.exists('./tmp' + '/' + model_name):
        os.makedirs('./tmp' + '/' + model_name)

    state_dict = model.state_dict()
    partial_dict = {k:v for (k,v) in state_dict.items() if 'layer.0.' in k or 'input' in k or 'output' in k}
    model_out_path = './tmp' + '/' + model_name + "/{}_base_epoch_{}.pth".format(model_name, epoch)

    torch.save(partial_dict, model_out_path)
    state = {"epoch": epoch ,"model": model}

    for i in range(model.module.nLayer - 3):
        partial_dict = {k:v for (k,v) in state_dict.items() if 'layer.%d.'%(i+1) in k}
        model_out_path = './tmp' + '/' + model_name + "/{}_res{}_epoch_{}.pth".format(model_name,i, epoch)
        torch.save(partial_dict, model_out_path)

    logging.info("Checkpoint saved components by components".format(model_out_path))
    #sys.exit()

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error)
        return loss

def get_criterion(loss_func):
    if loss_func == 'l2':
        criterion = nn.MSELoss(size_average=False)
    elif loss_func == 'l1_c':
        criterion = L1_Charbonnier_loss()
    else:
        print('invalid loss function')
        sys.exit()

    return criterion
