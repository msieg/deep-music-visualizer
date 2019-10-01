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

This is the only required argument! Example:

```bash
python deep_visualizer.py --song beethoven.mp3
```
