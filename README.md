# Deep Visualizer
The Deep Visualizer uses BigGAN (Brock et al., 2018) to visualize music.

Examples: https://www.instagram.com/deep_visualizer/

## Installation

This repo has been tested on Python3

Download this repository and run this command in terminal:

```bash
pip install -r requirements.txt
```

## How to run

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

### Pitch sensitivity

The pitch sensitivity controls how rapidly the class vector (thematic content of the video) will react to changes in pitch. The lower the number, the higher the sensitivity. 

Range: > 2
Recommended range: 5 - 200
Default: 50

Example:

```bash
python deep_visualizer.py --song beethoven.mp3 --pitch_sensitivity 10
```

### Tempo sensitivity

The tempo sensitivity controls how rapidly the noise vector (overall size, position, and orientation of objects) will react to changes in volume and tempo. The higher the number, the higher the sensitivity. 

Recommended range: 0.5 – 0.3
Default: 0.2

Example:

```bash
python deep_visualizer.py --song beethoven.mp3 --pitch_sensitivity 0.1
```

### Depth

This specifies the max value of the class vector units. Numbers closer to 1 yield more thematically rich content. Numbers closer to 0 seem to yield more 'deep' structures like human and dog faces. 

Range: 0.01 - 1
Default: 1

Example:

```bash
python deep_visualizer.py --song beethoven.mp3 --depth 0.5
```

### Classes

If you want to choose which classes (image categories) to visualize, you can specify a list of ImageNet indices (1-1000) here. ([list of ImageNet class indices](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)). The number of classes must be equal to [num_classes] (default is twelve, corresponding to the twelve musical pitches (A, A#, B, etc.)). You can also enter the class indices in order of priority (highest priority first) and set [sort_classes_by_power] to 1. 

Default: Twelve random indices between 1-1000

Example (if num_classes is set to default of twelve):
```bash
python deep_visualizer.py --song beethoven.mp3 --classes 45 99 567 234 89 90 105 998 56 677 884 530
```

### Num_classes

If you want to focus the visualizer around fewer than twelve themes, you can set num_classes to a number less than twelve. Since each class is associated with a pitch, the pitches that are kept when num_classes < 12 are those with the most overall power in the song. 

Default: 12

Example:
```bash
python deep_visualizer.py --song beethoven.mp3 --num_classes 4 
```

Or if you want to choose the classes:

```bash
python deep_visualizer.py --song beethoven.mp3 --num_classes 4 --classes 987 23 56 782
```

### Sort_classes_by_power

Set this to 1 if you want to prioritize the classes based on the order that you entered them in the [class] input. If you do not specify the [class] input, there is no reason to set this to 1. If you do specify the class input and do not set this to 1, the classes will be associated with the pitches in harmonic order from A, A#, B, etc.

Example:

```bash
python deep_visualizer.py --song beethoven.mp3  --classes 45 99 567 234 89 90 105 998 56 677 884 530 --sort_classes_by_power 1
```

### Jitter

The jitter prevents the same exact noise vectors from cycling repetitively during repetitive music so that the video output is more interesting. If you do want to cycle repetitively, set jitter to 0.

Range: 0 – 1
Default: 0.5 

Example:

```bash
python deep_visualizer.py --song beethoven.mp3 --jitter 0
```

### Frame_length

The frame length controls the number of audio frames per video frame in the output. If you want a very high quality video with a high frame rate, lower the frame_length. If you want a lower frame rate (perhaps if you are running on a CPU and want to cut down your runtime), raise the frame rate. The default of 512 is high quality. 

Range: Multiples of 2^6
Default: 512

Example:

```bash
python deep_visualizer.py --song beethoven.mp3 --frame_length 2048
```
