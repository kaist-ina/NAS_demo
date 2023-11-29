# NAS_demo
This repository holds the full end-to-end prototype for [Neural Adaptive Content-aware Internet Video Delivery (OSDI'18)](https://ina.kaist.ac.kr/~nas) with instructions for the demo.  
It consists of server-side HTML files, a client-side player, the content-aware SR DNN processor, and the Integrated ABR server.  
For the RL-based ABR such as NAS and Pensieve, we provide the pre-trained RL models.

## Project structure
```
./NAS_demo
├── dash.js                # JavaScript: modified DASH video player for NAS
├── dnn_processor          # Python: SR DNN inference processor
├── html                   # HTML: files for different schemes (NAS/Pensieve/robustMPC/BufferBased)
├── pensieve               # Python: ABR inference server with pre-trained RL models
├── sr_training            # Python: training code for content-aware SR DNN
├── super_resolution       # Python: NAS MDSR super_resolution models and tools
```

## Prerequisites
- Client (Player) OS: Ubuntu 18.04 or higher, Chrome Browser. (Any browser should work but only checked with Chrome).
- Client (Player) HW: NVIDIA GPU (2080 ti or better is recommended for real-time SR processing).
- Server (CDN): lighttpd or apache2 webserver.
- Dataset: DASH video chunks and NAS DNN Chunks. These are served at the CDN server. 

## Setup
### Client-side (Player)
- Clone repo and install conda env:
    ```
    git clone https://github.com/jaykim305/NAS_demo.git && cd NAS_demo && ./setup
    ```

- (Optional) Using Docker
    - Build docker
        ```
        sudo docker build -t nas-demo
        ```
    - Or you can pull our docker image `jaykim305/nas-demo:v1`. 
        The image is based on Ubuntu 18.04,  pytorch-cuda=11.8.  
        This might take some time due to the large image size.
        ```
        sudo docker pull jaykim305/nas-demo:v1
        ```                
    - Run docker and execute the rest of the instructions inside the docker.
        ```
        sudo docker run -it --gpus all -p 8333:8333 -p 5000:5000 -v $HOME/NAS_demo:/root/NAS_demo jaykim305/nas-demo:v1 /bin/bash
        ```
- Disable cache in Chrome browser. Refer to [this.](https://www.webinstinct.com/faq/how-to-disable-browser-cache#:~:text=When%20you're%20in%20Google,the%20box%20to%20Disable%20cache.)

### Server-side (CDN server)
In this setup, we provide the dataset [here.](https://www.dropbox.com/scl/fo/quk9mvt634ljii0nu0gux/h?rlkey=zvqbgbt65cqepjz2ni4fkvmpg&dl=0)   
Download and place it in your serving directory of your webserver. You can do this by following the instructions below.   
To use your own video and trained DNN for this demo, go to [Testing with your own video.](#testing-with-your-own-video)
- Install and start lighttpd (or apache2).
    ```
    sudo apt-get install lighttpd
    sudo systemctl start lighttpd
    sudo systemctl enable lighttpd
    sudo systemctl status lighttpd
    ```
- Download our dataset and copy it to ```/var/www/html```.
    ```
    sudo cp -r [dataset] /var/www/html/[your_dir_name]
    ```
    The structure should look like this:
    ```
    /var/www/html/[your_dir_name]
    ├── dash.all.debug.js                   # dash.js code
    ├── [content name]                      # DASH videos, DNNs, HTMLs
        ├── 1080p                 
        ├── 720p               
        ├── 480p           
        ├── 360p
        ├── 240p
        ├── ultra                           # NAS MDSR DNN Chunks
        ├── high                                
        ├── medium                            
        ├── low                               
        ├── multi_resolution_DNN.mpd        # MPD file
        ├── NAS_ultra.html                  # HTML file for different schemes
        ├── Pensieve.html
        ├── robustMPC.html
        ├── BufferBased.html 
    ```
- Replace the `<DNN url="<your webserver>/<your serving directory>/<content>">` in `multi_resolution_DNN.mpd` with your server's URL followed by the path to your content.

- Modify lighttpd config and restart.
    - Add `server.dir-listing = "enable"` in `/etc/lighttpd/lighttpd.conf`
    - Restart 
        ```
        sudo /etc/init.d/lighttpd restart
        ```
- (Optional) If you want the DNN processor/ABR server to run on a separate machine equipped with GPU, change DNN processing URL in the dash script.  
The default URL is set to the localhost.
    ```
    ./replace_url.sh dash.all.debug.js localhost <url of your processor> 
    ```
## Play Demo (Client-side)
You have the option to choose from the following schemes: **1) NAS, 2) Pensieve, 3) Robust MPC, 4) Buffer based**.
### Run ABR server 
```
cd ./pensieve
./run_abr.sh -t [type]
```
Options:
- `-t n`: Use NAS. Makes the decision to download either video or DNN chunk. It is RL-based and runs on TensorFlow.
- `-t p`: Use Pensieve ABR. It is RL-based and runs on TensorFlow.
- `-t m`: Use Robust MPC ABR.
- `-t s`: Use Buffer-based ABR.
- `-t r`: Replay video/dnn decision from a saved NAS trace. (Useful for debugging.)

### Run DNN processor (Only required for NAS)
```
cd ./dnn_processor
./dnn_server.sh -g [gpu device num] -c [content] -q [DNN quality] -d video
```
Options:
- `-g`: gpu device number
- `-q`: DNN quality. Choices: Low, Medium, High, Ultra. Refer to [NAS public repo](https://github.com/kaist-ina/NAS_public).

For the provided [dataset](https://www.dropbox.com/scl/fo/quk9mvt634ljii0nu0gux/h?rlkey=zvqbgbt65cqepjz2ni4fkvmpg&dl=0), we provide the ultra DNN quality. Therefore, you should set `-q utlra`.
### Play video with Chrome browser
Access the HTML webpage in your browser using the following address format: 
```
http://<your webserver>/<your serving directory>/<content>/<scheme>.html
```
Available schemes: `NAS_ultra.html, Pensieve.html, robustMPC.html, BuffBased.html`

## Tips
NAS can significantly enhance video quality in situations where network congestion occurs between the server and the client.  
To simulate such scenarios and observe the benefits of NAS, you can restrict the network speed on the server side.  
By doing so, you'll notice a noticeable improvement in video quality.  You can use the `tc` to limit network speed with the following rule:  
- Apply rule:
    ```
    sudo tc qdisc add dev eno1 root tbf rate 0.9mbit burst 32kbit latency 200ms
    ```
- Remove rule
    ```
    sudo tc qdisc del dev eno1 root
    ```
## Testing with your own video

### Setup
- Follow the setup in [Client-side (Player)](#client-side-player) to install required python packages.  
- Fetch code from [NAS public repo](https://github.com/kaist-ina/NAS_public). If you run the code below, the fetched code will be placed in the `sr_training`.
    ```
    git submodule update --init
    ```
### Prepare DASH video
- Download a video from YouTube using `yt-dlp`.
- Run script `./dash_vid_setup.sh` from [here.](https://github.com/kaist-ina/NAS_public#prepare-mpeg-dash-dataset)
- It will generate DASH video chunks and corresponding MPD file.
### Prepare DNN
- Train the content-aware DNNs. See the detailed instructions from [here.](https://github.com/kaist-ina/NAS_public#how-to-train-nas-mdsr)
    ```
    cd ./sr_training
    python train_nas_awdnn.py --quality [quality level] --data_name [dataset name] --use_cuda --load_on_memory
    ```
- (Optional) Generate the quality log. Required for evaluating effective bitrate and QoE.
    ```
    cd ./sr_training
    python test_nas_quality.py --quality [quality level] --fps 1 --data_name [content] --use_cuda --load_on_memory
    ```
### Add DNN config in MPD file
- Add the following DNN config in `mult_resolution.mpd` and save it as `multi_resolution_DNN.mpd`.  
An example MPD file can be found [here.](https://github.com/jaykim305/NAS_demo/blob/8d572007c23ae0f140fb43e05f86a5a706668ed2/html/multi_resolution_DNN.mpd#L24) Also, replace the `DNN url=` part with your server's URL.
```
 <DNN url="<your webserver>/<your serving directory>/<content>">
	<Representation id="low">
		<SegmentTemplate DNN="$RepresentaionID$/DNN_chunk_$Number$.pth" startNumber="1" endNumber="5"/>
	    <Quality id="240p" layer="10" feature="10"/>
		<Quality id="360p" layer="15" feature="40"/>
    </Representation>
	<Representation id="medium">
		<SegmentTemplate DNN="$RepresentaionID$/DNN_chunk_$Number$.pth" startNumber="1" endNumber="5"/>
        <Quality id="240p" layer="10" feature="24"/>
		<Quality id="360p" layer="15" feature="40"/>
	</Representation>	
	<Representation id="high">
		<SegmentTemplate DNN="$RepresentaionID$/DNN_chunk_$Number$.pth" startNumber="1" endNumber="5"/>
		<Quality id="240p" layer="10" feature="32"/>
		<Quality id="360p" layer="15" feature="40"/>
	</Representation>
 	<Representation id="ultra">
		<SegmentTemplate DNN="$RepresentaionID$/DNN_chunk_$Number$.pth" startNumber="1" endNumber="5"/>
        <Quality id="240p" layer="10" feature="44"/>
		<Quality id="360p" layer="15" feature="40"/>
	</Representation>
	<Representation id="single">
		<SegmentTemplate DNN="$RepresentaionID$/DNN_chunk_$Number$.pth" startNumber="1" endNumber="1"/>
	</Representation>
    <frameRate fps="24"/>
 </DNN>
```
### Upload & place all your content on the Server
```
./copy_files.sh // copies all necessary files to /var/www/html/[your serving directory]
```
### Play demo
Now you are ready. [Play video with Chrome browser](#play-video-with-chrome-browser).

### Changing schemes
- To change abr algorithm, modify `abr_id` in the html file.
    ```
    var abr_algorithms = {0: 'Default', 1: 'Fixed Rate (0)', 2: 'Buffer Based', 3: 'Rate Based', 4: 'RL', 5: 'Festive', 6: 'Bola'};
    var abr_id = 4;
    ```
- To change the DNN quality to download, modify `dnn_quality` in the html file.
    ```
    var dnn_quality = 3 // dnn_quality index: 0-low, 1-medium, 2-high, 3-ultra
    ```
- To enable/disable SR process, put `player.enableSRProcess()`/`player.disableSRProcess()` in the html file.
    ```
    player.enableSRProcess(); // enable
    player.disableSRProcess(); // disable
    ```


## Authors & Cite
- Hyunho Yeo, chaos5958@gmail.com
- Youngmok Jung, tom418@kaist.ac.kr
- Jaehong Kim, jaehong950305@gmail.com

If you use our work for research, please cite it.  
```
@inproceedings{yeo2018neural,
    title={Neural adaptive content-aware internet video delivery},
    author={Yeo, Hyunho and Jung, Youngmok and Kim, Jaehong and Shin, Jinwoo and Han, Dongsu},
    booktitle={13th $\{$USENIX$\}$ Symposium on Operating Systems Design and Implementation ($\{$OSDI$\}$ 18)},
    pages={645--661},
    year={2018}
}
```
### Commercial usage

* `BY-NC-SA` – [Attribution-NonCommercial-ShareAlike](https://github.com/idleberg/Creative-Commons-Markdown/blob/master/4.0/by-nc-sa.markdown)

NAS is currently protected under the patent and is retricted to be used for the commercial usage.

## Reference
- [Pensieve (SIGCOMM'17) repository](https://github.com/hongzimao/pensieve) for Pensieve ABR server.
- [NAS (OSDI'18) public repository](https://github.com/kaist-ina/NAS_public/tree/master) for NAS-MDSR module.
