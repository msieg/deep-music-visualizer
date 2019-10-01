import librosa
import librosa.display
import numpy as np
import moviepy.editor as mpy
import random
import torch

from PIL import Image,ImageDraw,ImageFont
from scipy.misc import toimage
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)


#choose model
#can also use -256 or -128 models to lower resolution and decrease runtime
model_name='biggan-deep-128'

# Load pre-trained model
model = BigGAN.from_pretrained(model_name)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#song file
song='lastdance.mp3'

#number of seconds of audio to visualize if you want to cuf off early
seconds=[[300]]

#if you want to repeat the same *starting* class and noise vectors, set to 1
use_last_vectors=0

#the theme rate controls the pitch sensitivity, i.e. how quickly the classes / themes will update based on changing pitch
#range: > 2
#numbers closer to 0 move the theme much faster
#recommended range: 5 to 200
theme_rate=40

#how "deep" do you want to travel into the latent space? 
#numbers closer to 1 keep you close to the surface of your imagenet classes
#numbers closer to 0 yield more latent structures (lots of dog and human faces)
#recommended range: 0.2 to 1
depth=1

#sensitivity of the noise vector to changes in tempo. recommended range: 0.1 to 0.2
tempo_sensitivity=0.2

#list of 12 imagenet class indices corresponding to 12 pitch class (A, A#, B, C, etc.)
#see 1000 imagenet class indices here: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
#if empty, 12 classes will be chosen randomly
classes=[]

#an integer between 1 and 12 indicating how many pitches / BigGAN classes to incorporate. 
#if less than 12, priority is given based on overall power in the song (see var chromasort)
#numbers closer to 1 yield more thematically simple videos.
num_pitches=12

#number of audio frames per video frame. 512 is standard in librosa. 
hop_length=512

#BigGAN noise vector truncation. 
#numbers closer to 1 yield more variable and unique image frames
truncation = 1

#jitter in the noise vector updates prevents frames from cycling repetiviely during repetitive audio
#if you do you want to cycle repetiively, set jitter to 0
jitter=0.5

#smoothing factor: smooth between the means of class vectors of this bin size
sm=10



########################################
########################################
########################################
########################################
########################################

print('\nReading audio \n')

#load song in librosa
y, sr = librosa.load(song)

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
if not classes:
    cls1000=list(range(1000))
    random.shuffle(cls1000)
    classes=cls1000[:12]


#initialize first class vector
cv1=np.zeros(1000)
for p in chromasort[:num_pitches]:
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


print('Generating input vectors \n')
for i in range(len(gradm)):   

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

    for j in range(num_pitches):

        #get class ind
        class_ind=classes[chromasort[j]]
        
        #get previous class unit 
        cva=cvlast[class_ind]

        #get current pitch power
        add = chroma[chromasort[j]][i] 
        
        #get current pitch gradient
        add2 = chromagrad[chromasort[j]][i]
  
        #update class unit based on theme_rate, current pitch power and gradient
        cv2[class_ind] = (cva + ((add+add2)/theme_rate))/(1+(1/(theme_rate/2)))



    

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
    cv2=cv2*depth
    
    #append new class vector
    class_vectors.append(cv2)
    
    #set last class vector
    cvlast=cv2





#interpolate between class vector bins to to smooth frames
class_vectors_terp=[]

for c in range(int(np.floor(len(class_vectors)/sm)-1)):

    ci=c*sm
        
    cva=np.mean(class_vectors[int(ci):int(ci)+sm],axis=0)
    cvb=np.mean(class_vectors[int(ci)+sm:int(ci)+sm*2],axis=0)
                
    for j in range(sm):
                                
        cvc = cva*(1-j/(sm-1)) + cvb*(j/(sm-1))
                                        
        class_vectors_terp.append(cvc)

class_vectors=np.array(class_vectors_terp)


if use_last_vectors==1:   
    class_vectors=np.load('class_vectors.npy')
    noise_vectors=np.load('noise_vectors.npy')
    
else:
    #save record of vectors for current video
    np.save('class_vectors.npy',class_vectors)
    np.save('noise_vectors.npy',noise_vectors)



#convert to Tensor
noise_vectors = torch.Tensor(np.array(noise_vectors))      
class_vectors = torch.Tensor(np.array(class_vectors))      



#Generate frames in batches of 30

batch_size=30

print('Generating frames \n')

#send to CUDA if running on GPU
model=model.to(device)
noise_vectors=noise_vectors.to(device)
class_vectors=class_vectors.to(device)




if isinstance(seconds,list)==False:
    frame_lim=int(np.floor(seconds*22050/512/batch_size))
else:
    frame_lim=999
    
frames = []
frame_ind=-batch_size

for i in range(frame_lim):
    
    frame_ind=i*batch_size

    print(str(np.round(frame_ind/len(noise_vectors)*100,2)) + '% complete')
    
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





