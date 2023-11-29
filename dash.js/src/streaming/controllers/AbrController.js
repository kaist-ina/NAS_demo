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

import SwitchRequest from '../rules/SwitchRequest';
import BitrateInfo from '../vo/BitrateInfo';
import DOMStorage from '../utils/DOMStorage';
import ABRRulesCollection from '../rules/abr/ABRRulesCollection';
import MediaPlayerModel from '../models/MediaPlayerModel';
import FragmentModel from '../models/FragmentModel';
import EventBus from '../../core/EventBus';
import Events from '../../core/events/Events';
import FactoryMaker from '../../core/FactoryMaker';
import ManifestModel from '../models/ManifestModel';
import DashManifestModel from '../../dash/models/DashManifestModel';
import VideoModel from '../models/VideoModel';
import DashMetrics from '../../dash/DashMetrics';
import MetricsModel from '../models/MetricsModel';
import PlaybackController from './PlaybackController';

const ABANDON_LOAD = 'abandonload';
const ALLOW_LOAD = 'allowload';
const DEFAULT_VIDEO_BITRATE = 1000;
const DEFAULT_AUDIO_BITRATE = 100;
const QUALITY_DEFAULT = 0;
//const dashMetrics = this.context.dashMetrics;
//const metricsModel = this.context.metricsModel;

function AbrController() {
    let context = this.context;
    let eventBus = EventBus(context).getInstance();
    let abrAlgo = -1;
    let bitrateArray = [200,300,480,750,1200,1850,2850,4300,5300];
    let reservoir = 5;
    let cushion = 10;
    let p_rb = 1;
    let pastThroughput = [];
    let pastDownloadTime = [];
    let bandwidthEstLog = [];
    let horizon = 5; // number of chunks considered
    let lastRequested = 0;
    window.lastQuality = 1;
    let lastQuality = 1;
    let alpha = 12;
    let qualityLog = [];
    let switchUpCount = 0;
    let switchUpThreshold = [0,1,2,3,4,5,6,7,8,9];
    let p = 0.85;
    let lastIndex = -1;
    let instance,
        abrRulesCollection,
        rulesController,
        streamController,
        autoSwitchBitrate,
        topQualities,
        qualityDict,
        confidenceDict,
        bitrateDict,
        ratioDict,
        averageThroughputDict,
        streamProcessorDict,
        abandonmentStateDict,
        abandonmentTimeout,
        limitBitrateByPortal,
        usePixelRatioInLimitBitrateByPortal,
        manifestModel,
        dashManifestModel,
        videoModel,
        dashMetrics,
        metricsModel,
        mediaPlayerModel,
        domStorage,
	playbackController;

    function setup() {
        autoSwitchBitrate = {video: true, audio: true};
        topQualities = {};
        qualityDict = {};
        confidenceDict = {};
        bitrateDict = {};
        ratioDict = {};
        averageThroughputDict = {};
        abandonmentStateDict = {};
        streamProcessorDict = {};
        limitBitrateByPortal = false;
        usePixelRatioInLimitBitrateByPortal = false;
        domStorage = DOMStorage(context).getInstance();
        mediaPlayerModel = MediaPlayerModel(context).getInstance();
        manifestModel = ManifestModel(context).getInstance();
        dashManifestModel = DashManifestModel(context).getInstance();
        videoModel = VideoModel(context).getInstance();
        dashMetrics = DashMetrics(context).getInstance();
        metricsModel = MetricsModel(context).getInstance();
	playbackController = PlaybackController(context).getInstance();
    }

    function initialize(type, streamProcessor) {
        streamProcessorDict[type] = streamProcessor;
        abandonmentStateDict[type] = abandonmentStateDict[type] || {};
        abandonmentStateDict[type].state = ALLOW_LOAD;
        eventBus.on(Events.LOADING_PROGRESS, onFragmentLoadProgress, this);


    }

    // returns size of last chunk using HTTPRequest object (not hardcoded :))
    function last_chunk_size(lastreq) {
        var tot = 0;
        for ( var tt = 0; tt < lastreq.trace.length; tt++ ) {
            tot = tot + lastreq.trace[tt]['b'][0];
        }
        return tot;
    }

    function next_chunk_size(index, quality) {
               // 9-bitrate weird video with 4 second chunks
       	if (index < 0 || index >48) {
            return 0;
        }
        
        return 0;
    }

    function getStabilityScore(b, b_ref, b_cur) {
        var score = 0,
        n = 0;
        if (lastIndex >= 1) {
            for (var i = Math.max(0, lastIndex + 1 - horizon); i<= lastIndex - 1; i++) {
            if (qualityLog[i] != qualityLog[i+1]) {
                n = n + 1;
            }
            }
        }
        if (b != b_cur) {
            n = n + 1;
        }
        score = Math.pow(2,n);
        return score;
    }

    function getEfficiencyScore(b, b_ref, w, bitrateArray) {
        var score = 0;
        score = Math.abs( bitrateArray[b]/Math.min(w, bitrateArray[b_ref]) - 1 );
        return score;
        }

        function getCombinedScore(b, b_ref, b_cur, w, bitrateArray) {
        var stabilityScore = 0,
        efficiencyScore = 0,
        totalScore = 0;
        // compute
        stabilityScore = getStabilityScore(b, b_ref, b_cur);
        efficiencyScore = getEfficiencyScore(b, b_ref, w, bitrateArray);
        totalScore = stabilityScore + alpha*efficiencyScore;
        return totalScore;  
    }

    function getBitrateFestive(prevQuality, bufferLevel, bwPrediction, lastRequested, bitrateArray) {
        var self = this, 
        bitrate = 0,
        tmpBitrate = 0,
        b_target = 0,
        b_ref = 0,
        b_cur = prevQuality,
        score_cur = 0,
        score_ref = 0;
        // TODO: implement FESTIVE logic
        // 1. log previous quality
        qualityLog[lastRequested] = prevQuality;
        lastIndex = lastRequested;
        // 2. compute b_target
        tmpBitrate = p*bwPrediction;
        for (var i = 9; i>=0; i--) { // todo: use bitrateArray.length
            if (bitrateArray[i] <= tmpBitrate) {
                b_target = i;
                break;
            }
            b_target = i;
        }
        // 3. compute b_ref
        if (b_target > b_cur) {
            switchUpCount = switchUpCount + 1;
            if (switchUpCount > switchUpThreshold[b_cur]) {
            b_ref = b_cur + 1;
            } else {
            b_ref = b_cur;
            }
        } else if (b_target < b_cur) {
            b_ref = b_cur - 1;
            switchUpCount = 0;
        } else {
            b_ref = b_cur;
            switchUpCount = 0; // this means need k consecutive "up" to actually switch up
        }
        // 4. delayed update
        if (b_ref != b_cur) { // need to switch
            // compute scores
            score_cur = getCombinedScore(b_cur, b_ref, b_cur, bwPrediction, bitrateArray);
            score_ref = getCombinedScore(b_ref, b_ref, b_cur, bwPrediction, bitrateArray);
            if (score_cur <= score_ref) {
            bitrate = b_cur;
            } else {
            bitrate = b_ref;
            if (bitrate > b_cur) { // clear switchupcount
                switchUpCount = 0;
            }
            }
        } else {
            bitrate = b_cur;
        }
        // 5. return
        return bitrate;
    }

    function predict_throughput(lastRequested, lastQuality, lastHTTPRequest) {
        var self = this,
        bandwidthEst = 0,
        lastDownloadTime,
        lastThroughput,
        lastChunkSize,
        tmpIndex,
        tmpSum = 0,
        tmpDownloadTime = 0;
        // First, log last download time and throughput
        if (lastHTTPRequest && lastRequested >= 0) {
            lastDownloadTime = (lastHTTPRequest._tfinish.getTime() - lastHTTPRequest.tresponse.getTime()) / 1000; //seconds
            if (lastDownloadTime <0.1) {
            lastDownloadTime = 0.1;
            }
            lastChunkSize = last_chunk_size(lastHTTPRequest);
            //lastChunkSize = self.vbr.getChunkSize(lastRequested, lastQuality);
            lastThroughput = lastChunkSize*8/lastDownloadTime/1000;
            // Log last chunk
            pastThroughput[lastRequested] = lastThroughput;
            pastDownloadTime[lastRequested] = lastDownloadTime;
        }
        // Next, predict future bandwidth
        if (pastThroughput.length === 0) {
            return 0;
        } else {
            tmpIndex = Math.max(0, lastRequested + 1 - horizon);
            tmpSum = 0;
            tmpDownloadTime = 0;
            for (var i = tmpIndex; i<= lastRequested; i++) {
            tmpSum = tmpSum + pastDownloadTime[i]/pastThroughput[i];
            tmpDownloadTime = tmpDownloadTime + pastDownloadTime[i];
            }
            bandwidthEst = tmpDownloadTime/tmpSum;
            bandwidthEstLog[lastRequested] = bandwidthEst;
            return bandwidthEst;
        }   
    }

    function setConfig(config) {
        if (!config) return;

        if (config.abrRulesCollection) {
            abrRulesCollection = config.abrRulesCollection;
        }
        if (config.rulesController) {
            rulesController = config.rulesController;
        }
        if (config.streamController) {
            streamController = config.streamController;
        }
    }

    function getBitrateBB(bLevel) {
        var self = this,
        tmpBitrate = 0,
        tmpQuality = 0;
        if (bLevel <= reservoir) {
            tmpBitrate = bitrateArray[0];
        } else if (bLevel > reservoir + cushion) {
            tmpBitrate = bitrateArray[8];
        } else {
            tmpBitrate = bitrateArray[0] + (bitrateArray[8] - bitrateArray[0])*(bLevel - reservoir)/cushion;
        }
        
        // findout matching quality level
        for (var i = 9; i>=0; i--) {
            if (tmpBitrate >= bitrateArray[i]) {
                tmpQuality = i;
                break;
            }
            tmpQuality = i;
        }
        //return 9;
        return tmpQuality;
        // return 0;
    }

    function getBitrateRB(bandwidth) {
        var self = this,
        tmpBitrate = 0,
        tmpQuality = 0;
        
        tmpBitrate = bandwidth*p_rb;
        
        // findout matching quality level
        for (var i = 9; i>=0; i--) {
            if (tmpBitrate >= bitrateArray[i]) {
                tmpQuality = i;
                break;
            }
            tmpQuality = i;
        }
        return tmpQuality;  
        // return 0;
    }

    function getTopQualityIndexFor(type, id) {
        var idx;
        topQualities[id] = topQualities[id] || {};

        if (!topQualities[id].hasOwnProperty(type)) {
            topQualities[id][type] = 0;
        }

        idx = checkMaxBitrate(topQualities[id][type], type);
        idx = checkMaxRepresentationRatio(idx, type, topQualities[id][type]);
        idx = checkPortalSize(idx, type);
        return idx;
    }

    /**
     * @param {string} type
     * @returns {number} A value of the initial bitrate, kbps
     * @memberof AbrController#
     */
    function getInitialBitrateFor(type) {

        let savedBitrate = domStorage.getSavedBitrateSettings(type);

        if (!bitrateDict.hasOwnProperty(type)) {
            if (ratioDict.hasOwnProperty(type)) {
                let manifest = manifestModel.getValue();
                let representation = dashManifestModel.getAdaptationForType(manifest, 0, type).Representation;

                if (Array.isArray(representation)) {
                    let repIdx = Math.max(Math.round(representation.length * ratioDict[type]) - 1, 0);
                    bitrateDict[type] = representation[repIdx].bandwidth;
                } else {
                    bitrateDict[type] = 0;
                }
            } else if (!isNaN(savedBitrate)) {
                bitrateDict[type] = savedBitrate;
            } else {
                bitrateDict[type] = (type === 'video') ? DEFAULT_VIDEO_BITRATE : DEFAULT_AUDIO_BITRATE;
            }
        }

        return bitrateDict[type];
    }

    /**
     * @param {string} type
     * @param {number} value A value of the initial bitrate, kbps
     * @memberof AbrController#
     */
    function setInitialBitrateFor(type, value) {
        bitrateDict[type] = value;
    }

    function getInitialRepresentationRatioFor(type) {
        if (!ratioDict.hasOwnProperty(type)) {
            return null;
        }

        return ratioDict[type];
    }

    function setInitialRepresentationRatioFor(type, value) {
        ratioDict[type] = value;
    }

    function getMaxAllowedBitrateFor(type) {
        if (bitrateDict.hasOwnProperty('max') && bitrateDict.max.hasOwnProperty(type)) {
            return bitrateDict.max[type];
        }
        return NaN;
    }

    //TODO  change bitrateDict structure to hold one object for video and audio with initial and max values internal.
    // This means you need to update all the logic around initial bitrate DOMStorage, RebController etc...
    function setMaxAllowedBitrateFor(type, value) {
        bitrateDict.max = bitrateDict.max || {};
        bitrateDict.max[type] = value;
    }

    function getMaxAllowedRepresentationRatioFor(type) {
        if (ratioDict.hasOwnProperty('max') && ratioDict.max.hasOwnProperty(type)) {
            return ratioDict.max[type];
        }
        return 1;
    }

    function setMaxAllowedRepresentationRatioFor(type, value) {
        ratioDict.max = ratioDict.max || {};
        ratioDict.max[type] = value;
    }

    function getAutoSwitchBitrateFor(type) {
        return autoSwitchBitrate[type];
    }

    function setAutoSwitchBitrateFor(type, value) {
        autoSwitchBitrate[type] = value;
    }

    function getLimitBitrateByPortal() {
        return limitBitrateByPortal;
    }

    function setLimitBitrateByPortal(value) {
        limitBitrateByPortal = value;
    }

    function getUsePixelRatioInLimitBitrateByPortal() {
        return usePixelRatioInLimitBitrateByPortal;
    }

    function setUsePixelRatioInLimitBitrateByPortal(value) {
        usePixelRatioInLimitBitrateByPortal = value;
    }

    function nextChunkQuality(buffer, lastRequested, lastQuality, rebuffer) {
        const metrics = metricsModel.getReadOnlyMetricsFor('video');
        //console.log("ORIG THROUGH: " + getAverageThroughput("video"));
        //var lastHTTPRequest = dashMetrics.getHttpRequests(metricsModel.getReadOnlyMetricsFor('video'))[lastRequested];
        var lastHTTPRequest = dashMetrics.getCurrentHttpRequest(metrics);
        var bandwidthEst = predict_throughput(lastRequested, lastQuality, lastHTTPRequest);
 
        switch(abrAlgo) {
            case 2:
                var xhr = new XMLHttpRequest();
		//jay mod for BB case, URL for abr server
                xhr.open("POST", "http://localhost:8333", false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                };

                var data = {'nextChunkSize': next_chunk_size(lastRequested+1), 'Type': 'BB', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                return getBitrateBB(buffer);
            case 3:
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "http://localhost:8333", false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                };
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1), 'Type': 'RB', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                return getBitrateRB(bandwidthEst);
            case 4:
                var quality = 2;
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "http://localhost:8333", false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText != "REFRESH" ) {
                            var resp = xhr.responseText.split("\t");
                            quality = parseInt(resp[0], 10);

                            if (resp.length == 2) {
                                window.reqByte = parseInt(resp[1], 10);
                            }
                        } else {
                            document.location.reload(true);
                        }
                    }
                };
                var bufferLevelAdjusted = buffer-0.15-0.4-4;
                var data = {'currentTime': new Date(), 'playTime': playbackController.getTime(), 'finished': 0, 'nextChunkSize': next_chunk_size(lastRequested+1), 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                
                if (lastQuality >= 5) {
                    var DNN_request = window.DNN_requests.pop();
                    data.lastChunkFinishTime = DNN_request.endTime.getTime();
                    data.lastChunkStartTime = DNN_request.startTime.getTime();
                    data.lastChunkSize = window.resByte;
                }

		        if (window.complete == 1) {
		            data.finished = 1;
                }

                xhr.send(JSON.stringify(data));
                console.log("QUALITY RETURNED IS: " + quality);
		
		        if (quality >= 5) {
		            window.reqType = "DNN";
		            //window.reqByte = 400000;
		        }
                else {
                    window.reqType = "video";
                }
                return quality;
            case 5:
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "http://localhost:8333", false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                };
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1), 'Type': 'Festive', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                var bufferLevelAdjusted = buffer-0.15-0.4-4;
                return getBitrateFestive(lastQuality, bufferLevelAdjusted, bandwidthEst, lastRequested, bitrateArray);
            case 6:
                /*
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "http://localhost:8333", false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                };
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1), 'Type': 'Bola', 'lastquality': lastQuality, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                data.finished = 0;
                xhr.send(JSON.stringify(data));*/
                return -1;
            default:
                // defaults to lowest quality always
                var xhr = new XMLHttpRequest();
                xhr.open("POST", "http://localhost:8333", false);
                xhr.onreadystatechange = function() {
                    if ( xhr.readyState == 4 && xhr.status == 200 ) {
                        console.log("GOT RESPONSE:" + xhr.responseText + "---");
                        if ( xhr.responseText == "REFRESH" ) {
                            document.location.reload(true);
                        }
                    }
                }
                var data = {'nextChunkSize': next_chunk_size(lastRequested+1), 'Type': 'Fixed Quality (0)', 'lastquality': 0, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                xhr.send(JSON.stringify(data));
                return 0;
        }
    }

    function getPlaybackQuality(streamProcessor, completedCallback, buffer=0, rebuffer=0) {
        const type = streamProcessor.getType();
        const streamInfo = streamProcessor.getStreamInfo();
        const streamId = streamInfo.id;

        const callback = function (res) {

            const topQualityIdx = getTopQualityIndexFor(type, streamId);

            let newQuality = res.value;

            /* if right before was DNN download */
            if (res.reason.name !== undefined && res.reason.name == 'NO_CHANGE' && window.DNN_download_enabled){
                newQuality = window.lastQuality;
            }

            if (newQuality < 0) {
                newQuality = 0;
            }
            if (newQuality > topQualityIdx) {
                newQuality = topQualityIdx;
            }

            const oldQuality = getQualityFor(type, streamInfo);
            if (newQuality !== oldQuality && (abandonmentStateDict[type].state === ALLOW_LOAD || newQuality > oldQuality)) {
                setConfidenceFor(type, streamId, res.confidence);
                changeQuality(type, streamInfo, oldQuality, newQuality, res.reason);
            }
            if (completedCallback) {
                completedCallback();
            }
        };

        //log("ABR enabled? (" + autoSwitchBitrate + ")");
        if (!getAutoSwitchBitrateFor(type)) {
            if (completedCallback) {
                completedCallback();
            }
        } else {
            const rules = abrRulesCollection.getRules(ABRRulesCollection.QUALITY_SWITCH_RULES);
            rulesController.applyRules(rules, streamProcessor, callback, getQualityFor(type, streamInfo), function (currentValue, newValue) {
                currentValue = currentValue === SwitchRequest.NO_CHANGE ? 0 : currentValue;
                if ( abrAlgo == 0 ) { // use the default return value
                    const metrics = metricsModel.getReadOnlyMetricsFor('video');
                    var lastHTTPRequest = dashMetrics.getCurrentHttpRequest(metrics);
                    var bandwidthEst = predict_throughput(lastRequested, lastQuality, lastHTTPRequest);
                    // defaults to lowest quality always
                    var xhr = new XMLHttpRequest();
                    xhr.open("POST", "http://localhost:8333", false);
                    xhr.onreadystatechange = function() {
                        if ( xhr.readyState == 4 && xhr.status == 200 ) {
                            console.log("GOT RESPONSE:" + xhr.responseText + "---");
                            if ( xhr.responseText == "REFRESH" ) {
                                document.location.reload(true);
                            }
                        }
                    }
                    var bufferLevelAdjusted = buffer-0.15-0.4-4;
                    var data = {'nextChunkSize': next_chunk_size(lastRequested+1), 'Type': 'Default', 'lastquality': 0, 'buffer': buffer, 'bufferAdjusted': bufferLevelAdjusted, 'bandwidthEst': bandwidthEst, 'lastRequest': lastRequested, 'RebufferTime': rebuffer, 'lastChunkFinishTime': lastHTTPRequest._tfinish.getTime(), 'lastChunkStartTime': lastHTTPRequest.tresponse.getTime(), 'lastChunkSize': last_chunk_size(lastHTTPRequest)};
                    xhr.send(JSON.stringify(data));
                    return Math.max(currentValue, newValue);
                }

                lastQuality = nextChunkQuality(buffer, lastRequested, window.lastQuality, rebuffer);
		        window.lastQuality = lastQuality;
                lastRequested = lastRequested + 1;
                window.lastRequested = lastRequested;
                if ( abrAlgo == 6 ) {
                    lastQuality = Math.max(currentValue, newValue);
                    window.lastQuality = lastQuality;
                    return Math.max(currentValue, newValue);
                }

                return lastQuality;
            });
        }
    }

    function setAbrAlgorithm(algo) {
        abrAlgo = algo;
        window.abrAlgo = abrAlgo;
    }

    function setPlaybackQuality(type, streamInfo, newQuality, reason) {
        const id = streamInfo.id;
        const oldQuality = getQualityFor(type, streamInfo);
        const isInt = newQuality !== null && !isNaN(newQuality) && (newQuality % 1 === 0);

        if (!isInt) throw new Error('argument is not an integer');

        if (newQuality !== oldQuality && newQuality >= 0 && newQuality <= getTopQualityIndexFor(type, id)) {
            changeQuality(type, streamInfo, oldQuality, newQuality, reason);
        }
    }

    function changeQuality(type, streamInfo, oldQuality, newQuality, reason) {
        setQualityFor(type, streamInfo.id, newQuality);
        eventBus.trigger(Events.QUALITY_CHANGE_REQUESTED, {mediaType: type, streamInfo: streamInfo, oldQuality: oldQuality, newQuality: newQuality, reason: reason});
    }


    function setAbandonmentStateFor(type, state) {
        abandonmentStateDict[type].state = state;
    }

    function getAbandonmentStateFor(type) {
        return abandonmentStateDict[type].state;
    }

    /**
     * @param {MediaInfo} mediaInfo
     * @param {number} bitrate A bitrate value, kbps
     * @returns {number} A quality index <= for the given bitrate
     * @memberof AbrController#
     */
    function getQualityForBitrate(mediaInfo, bitrate) {

        const bitrateList = getBitrateList(mediaInfo);

        if (!bitrateList || bitrateList.length === 0) {
            return QUALITY_DEFAULT;
        }

        for (let i = bitrateList.length - 1; i >= 0; i--) {
            const bitrateInfo = bitrateList[i];
            if (bitrate * 1000 >= bitrateInfo.bitrate) {
                return i;
            }
        }
        return 0;
    }

    /**
     * @param {MediaInfo} mediaInfo
     * @returns {Array|null} A list of {@link BitrateInfo} objects
     * @memberof AbrController#
     */
    function getBitrateList(mediaInfo) {
        if (!mediaInfo || !mediaInfo.bitrateList) return null;

        var bitrateList = mediaInfo.bitrateList;
        var type = mediaInfo.type;

        var infoList = [];
        var bitrateInfo;

        for (var i = 0, ln = bitrateList.length; i < ln; i++) {
            bitrateInfo = new BitrateInfo();
            bitrateInfo.mediaType = type;
            bitrateInfo.qualityIndex = i;
            bitrateInfo.bitrate = bitrateList[i].bandwidth;
            bitrateInfo.width = bitrateList[i].width;
            bitrateInfo.height = bitrateList[i].height;
            infoList.push(bitrateInfo);
        }

        return infoList;
    }

    function setAverageThroughput(type, value) {
        averageThroughputDict[type] = value;
    }

    function getAverageThroughput(type) {
        return averageThroughputDict[type];
    }

    function updateTopQualityIndex(mediaInfo) {
        var type = mediaInfo.type;
        var streamId = mediaInfo.streamInfo.id;
        var max = mediaInfo.representationCount - 1;

        setTopQualityIndex(type, streamId, max);

        return max;
    }

    function isPlayingAtTopQuality(streamInfo) {
        var isAtTop;
        var streamId = streamInfo.id;
        var audioQuality = getQualityFor('audio', streamInfo);
        var videoQuality = getQualityFor('video', streamInfo);

        isAtTop = (audioQuality === getTopQualityIndexFor('audio', streamId)) &&
            (videoQuality === getTopQualityIndexFor('video', streamId));

        return isAtTop;
    }

    function reset () {
        eventBus.off(Events.LOADING_PROGRESS, onFragmentLoadProgress, this);
        clearTimeout(abandonmentTimeout);
        abandonmentTimeout = null;
        setup();
    }

    function getQualityFor(type, streamInfo) {
        var id = streamInfo.id;
        var quality;

        qualityDict[id] = qualityDict[id] || {};

        if (!qualityDict[id].hasOwnProperty(type)) {
            qualityDict[id][type] = QUALITY_DEFAULT;
        }

        quality = qualityDict[id][type];
        return quality;
    }

    function setQualityFor(type, id, value) {
        qualityDict[id] = qualityDict[id] || {};
        qualityDict[id][type] = value;
    }

    function getConfidenceFor(type, id) {
        var confidence;

        confidenceDict[id] = confidenceDict[id] || {};

        if (!confidenceDict[id].hasOwnProperty(type)) {
            confidenceDict[id][type] = 0;
        }

        confidence = confidenceDict[id][type];

        return confidence;
    }

    function setConfidenceFor(type, id, value) {
        confidenceDict[id] = confidenceDict[id] || {};
        confidenceDict[id][type] = value;
    }

    function setTopQualityIndex(type, id, value) {
        topQualities[id] = topQualities[id] || {};
        topQualities[id][type] = value;
    }

    function checkMaxBitrate(idx, type) {
        var maxBitrate = getMaxAllowedBitrateFor(type);
        if (isNaN(maxBitrate) || !streamProcessorDict[type]) {
            return idx;
        }
        var maxIdx = getQualityForBitrate(streamProcessorDict[type].getMediaInfo(), maxBitrate);
        return Math.min (idx , maxIdx);
    }

    function checkMaxRepresentationRatio(idx, type, maxIdx) {
        var maxRepresentationRatio = getMaxAllowedRepresentationRatioFor(type);
        if (isNaN(maxRepresentationRatio) || maxRepresentationRatio >= 1 || maxRepresentationRatio < 0) {
            return idx;
        }
        return Math.min( idx , Math.round(maxIdx * maxRepresentationRatio) );
    }

    function checkPortalSize(idx, type) {
        if (type !== 'video' || !limitBitrateByPortal || !streamProcessorDict[type]) {
            return idx;
        }

        var hasPixelRatio = usePixelRatioInLimitBitrateByPortal && window.hasOwnProperty('devicePixelRatio');
        var pixelRatio = hasPixelRatio ? window.devicePixelRatio : 1;
        var element = videoModel.getElement();
        var elementWidth = element.clientWidth * pixelRatio;
        var elementHeight = element.clientHeight * pixelRatio;
        var manifest = manifestModel.getValue();
        var representation = dashManifestModel.getAdaptationForType(manifest, 0, type).Representation;
        var newIdx = idx;

        if (elementWidth > 0 && elementHeight > 0) {
            while (
                newIdx > 0 &&
                representation[newIdx] &&
                elementWidth < representation[newIdx].width &&
                elementWidth - representation[newIdx - 1].width < representation[newIdx].width - elementWidth
            ) {
                newIdx = newIdx - 1;
            }

            if (representation.length - 2 >= newIdx && representation[newIdx].width === representation[newIdx + 1].width) {
                newIdx = Math.min(idx, newIdx + 1);
            }
        }

        return newIdx;
    }

    function onFragmentLoadProgress(e) {
        const type = e.request.mediaType;
        if (getAutoSwitchBitrateFor(type)) {

            const rules = abrRulesCollection.getRules(ABRRulesCollection.ABANDON_FRAGMENT_RULES);
            const scheduleController = streamProcessorDict[type].getScheduleController();
            if (!scheduleController) return;// There may be a fragment load in progress when we switch periods and recreated some controllers.

            const callback = function (switchRequest) {
                if (switchRequest.confidence === SwitchRequest.STRONG &&
                    switchRequest.value < getQualityFor(type, streamController.getActiveStreamInfo())) {

                    const fragmentModel = scheduleController.getFragmentModel();
                    const request = fragmentModel.getRequests({state: FragmentModel.FRAGMENT_MODEL_LOADING, index: e.request.index})[0];
                    if (request) {
                        //TODO Check if we should abort or if better to finish download. check bytesLoaded/Total
                        fragmentModel.abortRequests();
                        setAbandonmentStateFor(type, ABANDON_LOAD);
                        setPlaybackQuality(type, streamController.getActiveStreamInfo(), switchRequest.value, switchRequest.reason);
                        eventBus.trigger(Events.FRAGMENT_LOADING_ABANDONED, {streamProcessor: streamProcessorDict[type], request: request, mediaType: type});

                        clearTimeout(abandonmentTimeout);
                        abandonmentTimeout = setTimeout(() => {
                            setAbandonmentStateFor(type, ALLOW_LOAD);
                            abandonmentTimeout = null;
                        }, mediaPlayerModel.getAbandonLoadTimeout());
                    }
                }
            };

            rulesController.applyRules(rules, streamProcessorDict[type], callback, e, function (currentValue, newValue) {
                return newValue;
            });
        }
    }

    instance = {
        isPlayingAtTopQuality: isPlayingAtTopQuality,
        updateTopQualityIndex: updateTopQualityIndex,
        getAverageThroughput: getAverageThroughput,
        getBitrateList: getBitrateList,
        getQualityForBitrate: getQualityForBitrate,
        getMaxAllowedBitrateFor: getMaxAllowedBitrateFor,
        setMaxAllowedBitrateFor: setMaxAllowedBitrateFor,
        getMaxAllowedRepresentationRatioFor: getMaxAllowedRepresentationRatioFor,
        setMaxAllowedRepresentationRatioFor: setMaxAllowedRepresentationRatioFor,
        getInitialBitrateFor: getInitialBitrateFor,
        setInitialBitrateFor: setInitialBitrateFor,
        getInitialRepresentationRatioFor: getInitialRepresentationRatioFor,
        setInitialRepresentationRatioFor: setInitialRepresentationRatioFor,
        setAutoSwitchBitrateFor: setAutoSwitchBitrateFor,
        getAutoSwitchBitrateFor: getAutoSwitchBitrateFor,
        setLimitBitrateByPortal: setLimitBitrateByPortal,
        getLimitBitrateByPortal: getLimitBitrateByPortal,
        getUsePixelRatioInLimitBitrateByPortal: getUsePixelRatioInLimitBitrateByPortal,
        setUsePixelRatioInLimitBitrateByPortal: setUsePixelRatioInLimitBitrateByPortal,
        getConfidenceFor: getConfidenceFor,
        getQualityFor: getQualityFor,
        getAbandonmentStateFor: getAbandonmentStateFor,
        setAbandonmentStateFor: setAbandonmentStateFor,
        setPlaybackQuality: setPlaybackQuality,
        setAbrAlgorithm: setAbrAlgorithm,
        getPlaybackQuality: getPlaybackQuality,
        nextChunkQuality: nextChunkQuality,
        setAverageThroughput: setAverageThroughput,
        getTopQualityIndexFor: getTopQualityIndexFor,
        initialize: initialize,
        setConfig: setConfig,
        reset: reset
    };

    setup();

    return instance;
}

AbrController.__dashjs_factory_name = 'AbrController';
let factory = FactoryMaker.getSingletonFactory(AbrController);
factory.ABANDON_LOAD = ABANDON_LOAD;
factory.QUALITY_DEFAULT = QUALITY_DEFAULT;
export default factory;
