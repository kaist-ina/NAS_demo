/**
 * The copyright in this software is being made available under the BSD License,
 * included below. This software may be subject to other third party and contributor
 * rights, including patent rights, and no such rights are granted under this license.
 *
 * Copyright (c) 2013, Dash Industry Forum.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation and/or
 *  other materials provided with the distribution.
 *  * Neither the name of Dash Industry Forum nor the names of its
 *  contributors may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
import {HTTPRequest} from '../vo/metrics/HTTPRequest';
import DataChunk from '../vo/DataChunk';
import FragmentModel from '../models/FragmentModel';
import MetricsModel from '../models/MetricsModel';
import EventBus from '../../core/EventBus';
import Events from '../../core/events/Events';
import FactoryMaker from '../../core/FactoryMaker';
import Debug from '../../core/Debug';
import PlaybackController from './PlaybackController';

function FragmentController(/*config*/) {

    const context = this.context;
    const log = Debug(context).getInstance().log;
    const eventBus = EventBus(context).getInstance();
    var playbackController = undefined;

    let instance,
        fragmentModels;

    function setup() {
        fragmentModels = {};
        eventBus.on(Events.FRAGMENT_LOADING_COMPLETED, onFragmentLoadingCompleted, instance);
        playbackController = PlaybackController(context).getInstance();
    }

    function process(bytes) {
        let result = null;
        if (bytes !== null && bytes !== undefined && bytes.byteLength > 0) {
            result = new Uint8Array(bytes);
        }
        return result;
    }

    function getModel(type) {
        let model = fragmentModels[type];
        if (!model) {
            model = FragmentModel(context).create({metricsModel: MetricsModel(context).getInstance()});
            fragmentModels[type] = model;
        }

        return model;
    }

    function isInitializationRequest(request) {
        return (request && request.type && request.type === HTTPRequest.INIT_SEGMENT_TYPE);
    }

    function reset() {
        eventBus.off(Events.FRAGMENT_LOADING_COMPLETED, onFragmentLoadingCompleted, this);
        for (let model in fragmentModels) {
            fragmentModels[model].reset();
        }
        fragmentModels = {};
    }

    function createDataChunk(bytes, request, streamId) {
        const chunk = new DataChunk();

        chunk.streamId = streamId;
        chunk.mediaInfo = request.mediaInfo;
        chunk.segmentType = request.type;
        chunk.start = request.startTime;
        chunk.duration = request.duration;
        chunk.end = chunk.start + chunk.duration;
        chunk.bytes = bytes;
        chunk.index = request.index;
        chunk.quality = request.quality;

        return chunk;
    }

    
    function send2DNNprocess(e) {
        if (window.SR_buffer.length == 0 && e === undefined) {
            return;
        }
        if (e === undefined){
            //var e = window.SR_buffer.shift();
            while (true){
                if (window.SR_buffer.length == 0) {
                    return;
                }
                var e = window.SR_buffer.shift();
                if (e.index * 4 > playbackController.getTime() + 4) {
                    break;
                }
            }
        }

        var xhr = new XMLHttpRequest();
        xhr.onload = function () {
            if (xhr.status >= 200 && xhr.status <= 299){
                console.log("GOT RESPONSE from DNN");
                var arraybuffer = xhr.response;
                if(arraybuffer.byteLength > 5){
                    var fname = xhr.getResponseHeader("Content-Disposition");   
                    eventBus.trigger(Events.MEDIA_FRAGMENT_LOADED, { name: fname, chunk: arraybuffer, fragmentModel: 'swap' });
                }
            }
        } 
        xhr.open("POST", "http://localhost:5000/uploader", true);
        xhr.responseType = "arraybuffer";
        var curTime = new Date();
        var playTime = playbackController.getTime();
        e.currentTime = curTime;
        e.playerTime = playTime;
        e.fps = window.fps;
        //xhr.setRequestHeader("Content-Type","multipart/form-data; boundary=----WebKitFormBoundaryF7SUcOWSQjdZfofk")
        let data = new FormData();
        data.append('videofile',new Blob([e.bytes], {type:"application/octet-stream"}));
        data.append('jsondata',JSON.stringify(e));
        xhr.send(data);
        console.log('-----------------------VIDEO SENT TO DNN PROCESSOR !!!: ' + e.index);
    }      


    function onFragmentLoadingCompleted(e) {
        if (fragmentModels[e.request.mediaType] !== e.sender) return;

        const scheduleController = e.sender.getScheduleController();
        const request = e.request;
        const bytes = e.response;
        const isInit = isInitializationRequest(request);
        const streamId = scheduleController.getStreamProcessor().getStreamInfo().id;

        if (!bytes) {
            log('No ' + request.mediaType + ' bytes to push.');
            return;
        }

        const chunk = createDataChunk(bytes, request, streamId);
        chunk.isInit = isInit;
        eventBus.trigger(isInit ? Events.INIT_FRAGMENT_LOADED : Events.MEDIA_FRAGMENT_LOADED, {chunk: chunk, fragmentModel: e.sender}); 

        if (window.SRProcess_enabled) {
            if (isInit) {
                send2DNNprocess(chunk);
            }
            else if (window.buffered > 4 && chunk.quality != 4) {
                window.SR_buffer.push(chunk);
            }
        }

    }

    instance = {
        process: process,
        getModel: getModel,
        isInitializationRequest: isInitializationRequest,
        reset: reset,
        send2DNNprocess: send2DNNprocess
    };

    setup();

    return instance;
}

FragmentController.__dashjs_factory_name = 'FragmentController';
export default FactoryMaker.getClassFactory(FragmentController);
