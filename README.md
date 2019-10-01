# Deep Visualizer
The Deep Visualizer uses BigGAN (Brock et al., 2018) to visualize music.

Examples: https://www.instagram.com/deep_visualizer/

# Installation

This repo has been tested on Python3

Download this repository and run this command in terminal:

```bash
pip install -r requirements.txt
```

# How to run

All features of the visualizer are available as input parameters. Each parameter is described below.

### song

Audio file of type mp3, wav, m4a, ogg, aac, au, or flac.

This is the only required argument!

Example:

```bash
python deep_visualizer.py --song beethoven.mp3
```

### model_name

biggan-deep-128, biggan-deep-256, or biggan-deep-512 (the number indicates the resolution of the output images, e.g. 128x128)

Default: biggan-deep-128

If you are running on a CPU (if you're not sure, you are on a CPU), you probably want to use biggan-deep-128 or else the code will take a very long time to run. Even with biggan-deep-128, this will take ~25 minutes to generate 1 minute of video on a MacBook Pro. We recommend using a virtual GPU on google cloud [link].

Example:

```bash
python deep_visualizer.py --song beethoven.mp3 --model_name biggan-deep-512
```
