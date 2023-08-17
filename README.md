# LaundryBot

## Setup & Installation
1. Clone this repo
2. Install all the things:
```
sudo apt install python3
sudo apt install python3-pip
pip install opencv-python
sudo apt install -y curl
curl -sL https://deb.nodesource.com/setup_14.x | sudo bash -
sudo apt install -y gcc g++ make build-essential nodejs sox gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps
npm config set user root && sudo npm install edge-impulse-linux -g --unsafe-perm
sudo apt install libgl1
pip install edgeimpulse-api
pip install edge_impulse_linux
sudo apt-get install portaudio19-dev
pip install pyaudio
pip install imutils
```
3. If using WSL, add USB support by following directions from here: <br />
https://learn.microsoft.com/en-us/windows/wsl/connect-usb
4. Download model file - Run the following command and follow the instructions:
```
edge-impulse-linux-runner
```


## Notes:
Based off of code from here:

https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning
