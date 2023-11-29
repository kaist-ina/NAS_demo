# Use the official Ubuntu 18.04 as the base image
FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y tmux ffmpeg libsm6 libxext6

# for abr and dnn server listening port
EXPOSE 8333 5000

RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
RUN shasum -a 256 Anaconda3-2023.09-0-Linux-x86_64.sh
RUN bash Anaconda3-2023.09-0-Linux-x86_64.sh -b
# RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

ENV PATH /root/anaconda3/bin:$PATH

# comment below lines if you want to install yourself for the specific pytorch cuda
# Copy your 'setup.sh' script from your local directory to the container 
COPY setup.sh .

# Make the 'setup.sh' script executable (if needed)
RUN chmod +x setup.sh

# Run the 'setup.sh' script
RUN ./setup.sh