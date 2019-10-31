import librosa
import argparse
import numpy as np
import moviepy.editor as mpy
import random
import torch
from PIL import Image
import imageio
from tqdm import tqdm
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample, convert_to_images)

#get input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", action="store_true", default=False)
parser.add_argument("-s", action="store_true", default=False)
parser.add_argument("--scaling", type=int, default=1)
parser.add_argument("--song", required=True)
parser.add_argument("--resolution", default='512')
parser.add_argument("--duration", type=int)
parser.add_argument("--pitch_sensitivity", type=int, default=220)
parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
parser.add_argument("--depth", type=float, default=1)
parser.add_argument("--classes", nargs='+', type=int)
parser.add_argument("--num_classes", type=int, default=12)
parser.add_argument("--sort_classes_by_power", type=int, default=0)
parser.add_argument("--jitter", type=float, default=0.5)
parser.add_argument("--frame_length", type=int, default=512)
parser.add_argument("--truncation", type=float, default=1)
parser.add_argument("--smooth_factor", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--use_previous_classes", type=int, default=0)
parser.add_argument("--use_previous_vectors", type=int, default=0)
parser.add_argument("--output_file", default="output.mp4")
args = parser.parse_args()

#if enabled uses disk to avoid storing all images in ram
save_only = args.s
use_disk = args.d

if use_disk == True:
    import os
    from os import listdir
    from os.path import isfile, join
    import shutil

    tmp_folder_path = os.path.join(os.getcwd(), "frames_output")

#read song
if args.song:
    song=args.song
    print('\nReading audio \n')
    y, sr = librosa.load(song)
else:
    raise ValueError("you must enter an audio file name in the --song argument")

#set model name based on resolution
model_name='biggan-deep-' + args.resolution
img_resolution = int(args.resolution)
frame_length=args.frame_length

#set pitch sensitivity
pitch_sensitivity=(300-args.pitch_sensitivity) * 512 / frame_length

#set tempo sensitivity
tempo_sensitivity=args.tempo_sensitivity * frame_length / 512

#set depth
depth=args.depth

#set number of classes  
num_classes=args.num_classes

#set sort_classes_by_power    
sort_classes_by_power=args.sort_classes_by_power

#set jitter
jitter=args.jitter
    
#set truncation
truncation=args.truncation

#set batch size  
batch_size=args.batch_size

#set use_previous_classes
use_previous_vectors=args.use_previous_vectors

#set use_previous_vectors
use_previous_classes=args.use_previous_classes
    
#set output name
outname=args.output_file

#set smooth factor
if args.smooth_factor > 1:
    smooth_factor=int(args.smooth_factor * 512 / frame_length)
else:
    smooth_factor=args.smooth_factor

#set duration  
if args.duration:
    seconds=args.duration
    frame_lim=int(np.floor(seconds*22050/frame_length/batch_size))
else:
    frame_lim=int(np.floor(len(y)/sr*22050/frame_length/batch_size))

if not save_only:
    # Load pre-trained model
    model = BigGAN.from_pretrained(model_name)

    #set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########################################
    ########################################
    ########################################
    ########################################
    ########################################

    #create spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000, hop_length=frame_length)

    #get mean power at each time point
    specm=np.mean(spec,axis=0)

    #compute power gradient across time points
    gradm=np.gradient(specm)

    #set max to 1
    gradm=gradm/np.max(gradm)

    #set negative gradient time points to zero 
    gradm = gradm.clip(min=0)
        
    #normalize mean power between 0-1
    specm=(specm-np.min(specm))/np.ptp(specm)

    #create chromagram of pitches X time points
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length)

    #sort pitches by overall power 
    chromasort=np.argsort(np.mean(chroma,axis=1))[::-1]

    ########################################
    ########################################
    ########################################
    ########################################
    ########################################

    if args.classes:
        classes=args.classes
        if len(classes) not in [12,num_classes]:
            raise ValueError("The number of classes entered in the --class argument must equal 12 or [num_classes] if specified")
        
    elif args.use_previous_classes==1:
        cvs=np.load('class_vectors.npy')
        classes=list(np.where(cvs[0]>0)[0])
        
    else: #select 12 random classes
        cls1000=list(range(1000))
        random.shuffle(cls1000)
        classes=cls1000[:12]
        



    if sort_classes_by_power==1:

        classes=[classes[s] for s in np.argsort(chromasort[:num_classes])]



    #initialize first class vector
    cv1=np.zeros(1000)
    for pi,p in enumerate(chromasort[:num_classes]):
        
        if num_classes < 12:
            cv1[classes[pi]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]       
        else:
            cv1[classes[p]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]

    #initialize first noise vector
    nv1 = truncated_noise_sample(truncation=truncation)[0]

    #initialize list of class and noise vectors
    class_vectors=[cv1]
    noise_vectors=[nv1]

    #initialize previous vectors (will be used to track the previous frame)
    cvlast=cv1
    nvlast=nv1


    #initialize the direction of noise vector unit updates
    update_dir=np.zeros(128)
    for ni,n in enumerate(nv1):
        if n<0:
            update_dir[ni] = 1
        else:
            update_dir[ni] = -1


    #initialize noise unit update
    update_last=np.zeros(128)

    ########################################
    ########################################
    ########################################
    ########################################
    ########################################

    #get new jitters
    def new_jitters(jitter):
        jitters=np.zeros(128)
        for j in range(128):
            if random.uniform(0,1)<0.5:
                jitters[j]=1
            else:
                jitters[j]=1-jitter        
        return jitters


    #get new update directions
    def new_update_dir(nv2,update_dir):
        for ni,n in enumerate(nv2):                  
            if n >= 2*truncation - tempo_sensitivity:
                update_dir[ni] = -1  
                            
            elif n < -2*truncation + tempo_sensitivity:
                update_dir[ni] = 1   
        return update_dir


    #smooth class vectors
    def smooth(class_vectors,smooth_factor):
        
        if smooth_factor==1:
            return class_vectors
        
        class_vectors_terp=[]
        for c in range(int(np.floor(len(class_vectors)/smooth_factor)-1)):  
            ci=c*smooth_factor          
            cva=np.mean(class_vectors[int(ci):int(ci)+smooth_factor],axis=0)
            cvb=np.mean(class_vectors[int(ci)+smooth_factor:int(ci)+smooth_factor*2],axis=0)
                        
            for j in range(smooth_factor):                                 
                cvc = cva*(1-j/(smooth_factor-1)) + cvb*(j/(smooth_factor-1))                                          
                class_vectors_terp.append(cvc)
                
        return np.array(class_vectors_terp)


    #normalize class vector between 0-1
    def normalize_cv(cv2):
        min_class_val = min(i for i in cv2 if i != 0)
        for ci,c in enumerate(cv2):
            if c==0:
                cv2[ci]=min_class_val    
        cv2=(cv2-min_class_val)/np.ptp(cv2) 
        
        return cv2

    print('\nGenerating input vectors \n')

    for i in tqdm(range(len(gradm))):   
        
        #print progress
        pass

        #update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
        if i%200==0:
            jitters=new_jitters(jitter)

        #get last noise vector
        nv1=nvlast

        #set noise vector update based on direction, sensitivity, jitter, and combination of overall power and gradient of power
        update = np.array([tempo_sensitivity for k in range(128)]) * (gradm[i]+specm[i]) * update_dir * jitters 
        
        #smooth the update with the previous update (to avoid overly sharp frame transitions)
        update=(update+update_last*3)/4
        
        #set last update
        update_last=update
            
        #update noise vector
        nv2=nv1+update

        #append to noise vectors
        noise_vectors.append(nv2)
        
        #set last noise vector
        nvlast=nv2
                    
        #update the direction of noise units
        update_dir=new_update_dir(nv2,update_dir)

        #get last class vector
        cv1=cvlast
        
        #generate new class vector
        cv2=np.zeros(1000)
        for j in range(num_classes):
            
            cv2[classes[j]] = (cvlast[classes[j]] + ((chroma[chromasort[j]][i])/(pitch_sensitivity)))/(1+(1/((pitch_sensitivity))))

        #if more than 6 classes, normalize new class vector between 0 and 1, else simply set max class val to 1
        if num_classes > 6:
            cv2=normalize_cv(cv2)
        else:
            cv2=cv2/np.max(cv2)
        
        #adjust depth    
        cv2=cv2*depth
        
        #this prevents rare bugs where all classes are the same value
        if np.std(cv2[np.where(cv2!=0)]) < 0.0000001:
            cv2[classes[0]]=cv2[classes[0]]+0.01

        #append new class vector
        class_vectors.append(cv2)
        
        #set last class vector
        cvlast=cv2

    #interpolate between class vectors of bin size [smooth_factor] to smooth frames 
    class_vectors=smooth(class_vectors,smooth_factor)

    #check whether to use vectors from last run
    if use_previous_vectors==1:   
        #load vectors from previous run
        class_vectors=np.load('class_vectors.npy')
        noise_vectors=np.load('noise_vectors.npy')
    else:
        #save record of vectors for current video
        np.save('class_vectors.npy',class_vectors)
        np.save('noise_vectors.npy',noise_vectors)

    ########################################
    ########################################
    ########################################
    ########################################
    ########################################

    #convert to Tensor
    noise_vectors = torch.Tensor(np.array(noise_vectors))      
    class_vectors = torch.Tensor(np.array(class_vectors))      


    #Generate frames in batches of batch_size
    print('\n\nGenerating frames \n')

    #send to CUDA if running on GPU
    model=model.to(device)
    noise_vectors=noise_vectors.to(device)
    class_vectors=class_vectors.to(device)

    #Creating temporary directory or init a frames list
    if use_disk == True:
        if os.path.exists(tmp_folder_path):
            shutil.rmtree(tmp_folder_path)
        os.mkdir(tmp_folder_path)
        counter = 0
    else:
        frames = []

    for i in tqdm(range(frame_lim)):
        
        #print progress
        pass

        if (i+1)*batch_size > len(class_vectors):
            torch.cuda.empty_cache()
            break
        
        #get batch
        noise_vector=noise_vectors[i*batch_size:(i+1)*batch_size]
        class_vector=class_vectors[i*batch_size:(i+1)*batch_size]

        # Generate images
        with torch.no_grad():
            output = model(noise_vector, class_vector, truncation)

        output_cpu = output.cpu()
        pil_imgs = convert_to_images(output_cpu)

        #convert to image array and add to frames
        for out in pil_imgs:
            if args.scaling != 1:
                newsize = (img_resolution * args.scaling,
                           img_resolution * args.scaling)
                im = np.array(out.resize(newsize, Image.LANCZOS))
            else:
                im = np.array(out)
            if use_disk == True:
                imageio.imwrite(os.path.join(
                    tmp_folder_path, str(counter) + ".png"), im)
                counter=counter + 1
            else:
                frames.append(im)

        #empty cuda cache
        torch.cuda.empty_cache()

#Save video
try:
    aud = mpy.AudioFileClip(song, fps = 44100) 

    if args.duration:
        aud.duration=args.duration

    if use_disk == True:
        files_path = [os.path.join(tmp_folder_path, x)
                    for x in os.listdir(tmp_folder_path) if x.endswith('.png')]
        files_path.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        clip = mpy.ImageSequenceClip(files_path, fps=22050/frame_length)
        clip = clip.set_audio(aud)
        clip.write_videofile(outname, audio_codec='aac')
    else:
        clip = mpy.ImageSequenceClip(frames, fps=22050/frame_length)
        clip = clip.set_audio(aud)
        clip.write_videofile(outname, audio_codec='aac')

    # Removing tmp directory
    if use_disk == True and not save_only:
        print("\nCleaning tmp directory\n")
        if os.path.exists(tmp_folder_path):
            shutil.rmtree(tmp_folder_path)
except Exception as e:
    print("Failed to save the file")
