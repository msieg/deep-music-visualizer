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

### Song

Audio file of type mp3, wav, m4a, ogg, aac, au, or flac.

This is the only required argument!

Example:

```bash
python deep_visualizer.py --song beethoven.mp3
```

### Resolution

128, 256, or 512

Default: 128

If you are running on a CPU (if you're not sure, you are on a CPU), you probably want to use a resolution of 128 or else the code will take a very long time to run. Even at 128, this will take ~25 minutes to generate 1 minute of video on a MacBook Pro. We recommend using a virtual GPU on google cloud [link] with a resolution of 512.

Example:

```bash
python deep_visualizer.py --song beethoven.mp3 --resolution 512
```

### Duration

Duration of the video output, in seconds.

Default: Full length of the audio

Example:

```bash
python deep_visualizer.py --song beethoven.mp3 --duration 30
```
