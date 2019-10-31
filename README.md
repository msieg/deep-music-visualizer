# Deep Music Visualizer
The Deep Music Visualizer uses BigGAN (Brock et al., 2018), a generative neural network, to visualize music. Like this:

[![Alt text](https://img.youtube.com/vi/L7R-yBZ5QYc/0.jpg)](https://www.youtube.com/watch?v=L7R-yBZ5QYc)

More examples: https://www.instagram.com/deep_music_visualizer/

## Installation

This repo has been tested on Python3

Assuming you have python installed, open terminal and run these commands:

```bash
git clone https://github.com/msieg/deep-music-visualizer.git
cd deep-music-visualizer
pip install -r requirements.txt
```

If you are on linux, you may also need to run:

```bash
apt-get update
apt-get install ffmpeg
apt-get install libsndfile1
```


## How to run

All features of the visualizer are available as input parameters. Each parameter and option are described below.

## Options

### -d: use disk

Store the generated images to disk during the generation process, this option reduces significantly the cpu memory footprint.

Example:

```bash
python visualize.py -d --song beethoven.mp3
```

### -s: save only

Read from the temporary folder (created only if the software previously runned in *use disk* mode) and creates the video output. This option is quite usefull if Moviepy stucks during the video writing process.

Example:

```bash
python visualize.py -s --song beethoven.mp3
```

## Parameters

### --song: input song

Audio file of type mp3, wav, or ogg.

This is the only required argument!

Example:

```bash
python visualize.py --song beethoven.mp3
```

### --resolution: GAN resolution

If you are running on a CPU (if you're not sure, you are on a CPU), you might want to use a lower resolution or else the code will take a very long time to run. At 512X512, it will take ~7 HOURS to generate 1 minute of video on a standard desktop computer (assuming all other parameters are default). At 128x128, it would take ~25 minutes. To speed up runtime, you can decrease the resolution or increase the [frame_length](#Frame_length). To dramatically speed up runtime and generate higher quality videos, use a resolution of 512 on a [GPU on a google cloud virtual machine](https://cloud.google.com/deep-learning-vm/docs/cloud-marketplace).

Default: 512 (128, 256, or 512)

Example:

```bash
python visualize.py --song beethoven.mp3 --resolution 128
```

### --duration: videoclip duration

Duration of the video output in seconds. It can be useful to generate shorter videos while you are tweaking the other visualizer parameters. Once you find your preferred parameters, remove the duration argument and set [use_previous_vectors](#use_previous_vectors) to 1 to generate the same video but for the full duration of the song. 

Default: Full length of the audio

Example:

```bash
python visualize.py --song beethoven.mp3 --duration 30
```

### --pitch_sensitivity: pitch sensitivity

The pitch sensitivity controls how rapidly the class vector (thematic content of the video) will react to changes in pitch. The higher the number, the higher the sensitivity. 

Range: 1 – 299

Recommended range: 200 – 295

Default: 220

Example:

```bash
python visualize.py --song beethoven.mp3 --pitch_sensitivity 280
```

### --tempo_sensitivity: tempo sensitivity

The tempo sensitivity controls how rapidly the noise vector (i.e. the overall size, position, and orientation of objects in the images) will react to changes in volume and tempo. The higher the number, the higher the sensitivity. 

Recommended range: 0.05 – 0.8

Default: 0.25

Example:

```bash
python visualize.py --song beethoven.mp3 --tempo_sensitivity 0.1
```

### --depth: depth

The depth specifies the max value of the class vector units. Numbers closer to 1 seem to yield more thematically rich content. Numbers closer to 0 seem to yield more 'deep' structures like human and dog faces. However, this depends heavily on the specific classes you are using.  

Range: 0.01 - 1

Default: 1

Example:

```bash
python visualize.py --song beethoven.mp3 --depth 0.5
```

### --classes: visualized classes

If you want to choose which classes (image categories) to visualize, you can specify a list of ImageNet indices (1-1000) here. ([list of ImageNet class indices](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)). The number of classes must be equal to [num_classes] (default is twelve, corresponding to the twelve musical pitches (A, A#, B, etc.)). You can also enter the class indices in order of priority (highest priority first) and set [sort_classes_by_power](#sort_classes_by_power) to 1. 

Default: Twelve random indices between 0-999

Example (if num_classes is set to default of twelve):
```bash
python visualize.py --song beethoven.mp3 --classes 45 99 567 234 89 90 105 998 56 677 884 530
```

### --num_classes: number of used classes

If you want to focus the visualizer around fewer than twelve themes, you can set num_classes to a number less than twelve. Since each class is associated with a pitch, the pitches that are retained when num_classes <= 12 are those with the most overall power in the song. 

Default: 12 (<= 12)

Example:
```bash
python visualize.py --song beethoven.mp3 --num_classes 4 
```

Or if you want to choose the classes:

```bash
python visualize.py --song beethoven.mp3 --num_classes 4 --classes 987 23 56 782
```

### --sort_classes_by_power: class order priority

Set this to 1 if you want to prioritize the classes based on the order that you entered them in the [class input](#classes). If you do not specify the class input, there is no reason to set this to 1. If you do specify the class input and do not set this to 1, the classes will be associated with the pitches in harmonic order from A, A#, B, etc.

Example:

```bash
python visualize.py --song beethoven.mp3  --classes 45 99 567 234 89 90 105 998 56 677 884 530 --sort_classes_by_power 1
```

### --jitter: jitter

The jitter prevents the same exact noise vectors from cycling repetitively during repetitive music so that the video output is more interesting. If you do want to cycle repetitively, set jitter to 0.

Range: 0 – 1
Default: 0.5 

Example:

```bash
python visualize.py --song beethoven.mp3 --jitter 0
```

### --frame_length: audio and video frame ratio

The frame length controls the number of audio frames per video frame in the output. If you want a higher frame rate for visualizing very rapid music, lower the frame_length. If you want a lower frame rate (perhaps if you are running on a CPU and want to cut down your runtime), raise the frame_length. The default of 512 is high quality. 

Range: Multiples of 2^6

Default: 512

Example:

```bash
python visualize.py --song beethoven.mp3 --frame_length 2048
```

### --truncation: image generation variability

The truncation controls the variability of images that BigGAN generates by limiting the max values in the noise vector. Truncations closer to 1 yield more variable images, and truncations closer to 0 yield simpler images with more recognizable, normal-looking objects. 

Range: 0.1 - 1

Default: 1

Example:

```bash
python visualize.py --song beethoven.mp3 --truncation 0.4
```

### --smooth_factor: pitch fluctuation reduction

After the class vectors have been generated, they are smoothed by interpolating linearly between the means of class vectors in bins of size [smooth_factor]. This is performed because small local fluctuations in pitch can cause the video frames to fluctuate back and forth. If you want to visualize very fast music with rapid changes in pitch, you can lower the smooth factor. You may also want to lower the frame_length in that case. However, for most songs, it is difficult to avoid rapid fluctuations with smooth factors less than 10. 

Range: > 0

Recommended range: 10 – 30

Default: 20

Example:

```bash
python visualize.py --song beethoven.mp3 --smooth_factor 6
```

### --batch_size: GAN batch size

BigGAN generates the images in batches of size [batch_size]. The only reason to reduce batch size from the default of 30 is if you run out of CUDA memory on a GPU. Reducing the batch size will slightly increase overall runtime. 

Default: 30

Example:

```bash
python visualize.py --song beethoven.mp3 --batch_size 20
```

### --use_previous_classes: reuse previous random classes

If your previous run of the visualizer used random classes (i.e. you did not manually set the [class](#classes) input), and you liked the video output but want to mess with some other parameters, set use_previous_classes to 1 so that you create a similar video with the same classes on the next run of the code. 

Default: 0

Example:

```bash
python visualize.py --song beethoven.mp3 --use_previous_classes 1
```

### --use_previous_vectors: use_previous_vectors

If you're messing around with the visualizer parameters, it can be useful to generate videos with short durations to keep your runtime low. Once you find the right set of parameters, remove the [duration](#duration) argument and set use_previous_vectors to 1 to generate the same video again, but for a longer duration. 

Default: 0

Example:

```bash
python visualize.py --song beethoven.mp3 --use_previous_vectors 1
```

### --output_file: output filename

Output file name

Default: output.mp4

Example:

```bash
python visualize.py --song beethoven.mp3 --output_file my_movie.mp4
```

### --scaling: image resampling

Positive integer which allows to scale the the generated images (Lanczos resampling). The resampling process will slow down the generation process.

Default: 1 (> 0)

Example:

```bash
python visualize.py --song beethoven.mp3 --scaling 2
```