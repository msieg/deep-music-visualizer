import librosa
import argparse
import numpy as np
import moviepy.editor as mpy
import random
import torch

from scipy.misc import toimage
from tqdm import tqdm
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

sys.exit()

parser = argparse.ArgumentParser()

parser.add_argument("--song")
parser.add_argument("--model_name")
parser.add_argument("--duration")
parser.add_argument("--pitch_sensitivity")
parser.add_argument("--tempo_sensitivity")
parser.add_argument("--depth")
parser.add_argument("--classes")
parser.add_argument("--num_classes")
parser.add_argument("--jitter")
parser.add_argument("--hop_length")
parser.add_argument("--truncation")
parser.add_argument("--smooth_factor")
parser.add_argument("--batch_size")
parser.add_argument("--use_previous_classes")
parser.add_argument("--use_previous_vectors")

args = parser.parse_args()

if args.song:
    song=args.song
    y, sr = librosa.load(song)
else:
    raise ValueError("you must enter an audio file name in the --song argument")

if args.model_name:
    model_name=args.model_name
else:
    model_name='biggan-deep-512'
    
if args.pitch_sensitivity:
    pitch_sensitivity=args.pitch_sensitivity
else:
    pitch_sensitivity=50  
    
if args.tempo_sensitivity:
    tempo_sensitivity=args.tempo_sensitivity
else:
    tempo_sensitivity=0.2
    
if args.depth:
    depth=args.depth
else:
    depth=0
   
if args.num_classes:
    num_classes=args.num_classes
else:
    num_classes=12

if args.jitter:
    jitter=args.jitter
else:
    jitter=0.5

if args.hop_length:
    hop_length=args.hop_length
else:
    hop_length=512
    
if args.truncation:
    truncation=args.truncation
else:
    truncation=1
    
if args.smooth_factor:
    smooth_factor=args.smooth_factor
else:
    smooth_factor=10
    
if args.batch_size:
    batch_size=args.batch_size
else:
    batch_size=30
    
if args.duration:
    seconds=args.duration
    frame_lim=int(np.floor(seconds*22050/hop_length/batch_size))
else:
    frame_lim=int(np.floor(len(y)/sr*22050/hop_length/batch_size))
    
if args.use_previous_vectors:
    use_previous_vectors=args.use_previous_vectors
else:
    use_previous_vectors=0



# Load pre-trained model
model = BigGAN.from_pretrained(model_name)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




########################################
########################################
########################################
########################################
########################################

print('\nReading audio \n')

#load song in librosa


#create spectrogram
spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000, hop_length=hop_length)

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
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

#compute power gradient for each pitch / class
chromagrad=np.gradient(chroma)[1]

#sort pitches by overall power 
chromasort=np.argsort(np.mean(chroma,axis=1))[::-1]


########################################
########################################
########################################
########################################
########################################


#If class list is empty, select 12 random classes
if args.classes:
    classes=args.classes
else:
    cls1000=list(range(1000))
    random.shuffle(cls1000)
    classes=cls1000[:12]
    


#initialize first class vector
cv1=np.zeros(1000)
for p in chromasort[:num_classes]:
    cv1[classes[p]] = chroma[p][0]


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
        

#initialize noise unit jitters
jitters=np.zeros(128)


#initialize noise unit update
update_last=np.zeros(128)


########################################
########################################
########################################
########################################
########################################


print('\n\nGenerating input vectors \n')

for i in tqdm(range(len(gradm))):   
    
    pass

    #update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
    if i%100==0:
        for j in range(128):
            if random.uniform(0,1)<0.5:
                jitters[j]=1
            else:
                jitters[j]=1-jitter

    
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
    for ni,n in enumerate(nv2):
                     
        if n >= 2*truncation - (tempo_sensitivity+0.5):
            update_dir[ni] = -1  
                        
        elif n < -2*truncation + (tempo_sensitivity+0.5):
            update_dir[ni] = 1
                



    

    #get last class vector
    cv1=cvlast
    
    #initialize new class vector
    cv2=np.zeros(1000)

    for j in range(num_classes):

        #get class ind
        class_ind=classes[chromasort[j]]
        
        #get previous class unit 
        cva=cvlast[class_ind]

        #get current pitch power
        add = chroma[chromasort[j]][i] 
        
        #get current pitch gradient
        add2 = chromagrad[chromasort[j]][i]
  
        #update class unit based on pitch_sensitivity, current pitch power and gradient
        cv2[class_ind] = (cva + ((add+add2)/pitch_sensitivity))/(1+(1/(pitch_sensitivity/2)))



    

    #normalize new class vector between 0 and 1
    min_class_val = min(i for i in cv2 if i != 0)
    for ci,c in enumerate(cv2):
        if c==0:
            cv2[ci]=min_class_val    
    cv2=(cv2-min_class_val)/np.ptp(cv2) 
    
    
    #this prevents rare bugs where all classes are the same value
    if np.std(cv2[np.where(cv2!=0)]) < 0.0000001:
        cv2[chromasort[0]]=cv2[chromasort[0]]+0.01

    #adjust depth    
    cv2=cv2*(1-depth)
    
    #append new class vector
    class_vectors.append(cv2)
    
    #set last class vector
    cvlast=cv2





#interpolate between class vector bins to to smooth frames
class_vectors_terp=[]

for c in range(int(np.floor(len(class_vectors)/smooth_factor)-1)):

    ci=c*smooth_factor
        
    cva=np.mean(class_vectors[int(ci):int(ci)+smooth_factor],axis=0)
    cvb=np.mean(class_vectors[int(ci)+smooth_factor:int(ci)+smooth_factor*2],axis=0)
                
    for j in range(smooth_factor):
                                
        cvc = cva*(1-j/(smooth_factor-1)) + cvb*(j/(smooth_factor-1))
                                        
        class_vectors_terp.append(cvc)

class_vectors=np.array(class_vectors_terp)


if use_previous_vectors==1:   
    #load vectors from previous run
    class_vectors=np.load('class_vectors.npy')
    noise_vectors=np.load('noise_vectors.npy')
    
else:
    #save record of vectors for current video
    np.save('class_vectors.npy',class_vectors)
    np.save('noise_vectors.npy',noise_vectors)



#convert to Tensor
noise_vectors = torch.Tensor(np.array(noise_vectors))      
class_vectors = torch.Tensor(np.array(class_vectors))      



#Generate frames in batches of batch_size

print('\n\nGenerating frames \n')

#send to CUDA if running on GPU
model=model.to(device)
noise_vectors=noise_vectors.to(device)
class_vectors=class_vectors.to(device)


frames = []


for i in tqdm(range(frame_lim)):
    
    pass

    if (i+1)*batch_size > len(class_vectors):
        break
    
    noise_vector=noise_vectors[i*batch_size:(i+1)*batch_size]
    class_vector=class_vectors[i*batch_size:(i+1)*batch_size]

    # Generate images
    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)

    output_cpu=output.cpu().data.numpy()

    for out in output_cpu:    
     
        im=np.array(toimage(out))

        frames.append(im)
        
    
    torch.cuda.empty_cache()



#Save video  
aud = mpy.AudioFileClip(song,fps = 44100) 
clip = mpy.ImageSequenceClip(frames, fps=22050/hop_length)
clip = clip.set_audio(aud)
clip.write_videofile("output.mp4",audio_codec='aac')





