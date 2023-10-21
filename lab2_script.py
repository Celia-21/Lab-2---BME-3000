# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 19:25:04 2023

@author: celia
"""

#lab2_script.py
#Script for lab 3 that explores the result of convolution and designing different filters to alter sound quality

#import packages
import numpy as np
from matplotlib import pyplot as plt
import lab2_module as L2
import soundfile as sf
import sounddevice as sd

#%% Part 1
#declare variables
dt=0.01
#create a time vector 
time=np.arange(0,5.01,dt)
#create variable that models the provided input signal, x(t)
input_signal=np.sin(np.pi*6*time)

#create system impulse function
system_impulse=np.zeros(len(time))
system_impulse[(time>0.5)&(time<2)]=1
system_impulse[(time<0.5)&(time>2)]=0

#create variable input_signal_scaled
input_signal_scaled=2*input_signal
#create a figure
plt.figure(1,clear=True)

#Row 0
#create subplot 1
plt.subplot(3,3,1)
plt.plot(time,input_signal)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')
#create subplot 2
plt.subplot(3,3,2)
plt.plot(time,system_impulse)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')
#create subplot 3 for the convolution
plt.subplot(3,3,3)
convolved_input_impulse=np.convolve(input_signal, system_impulse)
convolve_time_array=np.arange(len(convolved_input_impulse))*dt
plt.plot(convolve_time_array,convolved_input_impulse)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')

#Row 1
#create subplot 4
plt.subplot(3,3,4)
plt.plot(time,system_impulse)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')
#create sub 5
plt.subplot(3,3,5)
plt.plot(time,input_signal)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')
#create sub 6 for the convolution
plt.subplot(3,3,6)
convolved_impulse_input=np.convolve(system_impulse,input_signal)
plt.plot(convolve_time_array,convolved_impulse_input)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')

#Row 2
#create subplot 7
plt.subplot(3,3,7)
plt.plot(time,input_signal_scaled)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')
#create subplot 8
plt.subplot(3,3,8)
plt.plot(time,system_impulse)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')
#create subplot 9 for the convolution
plt.subplot(3,3,9)
convolved_input_system=np.convolve(input_signal_scaled, system_impulse)
plt.plot(convolve_time_array,convolved_input_system)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')

#annotate with title and fix layout
plt.suptitle('Visualization of Convolution Properties')
plt.tight_layout()

#print statement of the convolution properties
print("Observed in the subplot, we see that Communativity and Distributivity Associativity were properties used.\n Commutativity means that the order of the two operands does not impact the result.\n Associativity means that the results of an operation in series are the same regardless of the order. \n Distributivity means that two signals can be combined, then convolved to get the same result as convolving each then combining.")
plt.savefig("convolved_signal_graphs.pdf")

#%% Part 2
#call function from module
my_convolved_signal = L2.get_convolved_signal(input_signal, system_impulse)

#create plot
plt.figure(2,clear=True)
plt.plot(convolve_time_array, my_convolved_signal)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U)')
plt.title('Input Signal Convolved with System Impulse')
plt.savefig("convolved_arrays.pdf")

#check if np.convolve produces the same result
is_arrays_same = np.array_equal(my_convolved_signal, convolved_impulse_input)
if is_arrays_same == 1:
    is_is_not = 'is'
else:
    is_is_not = "is not"
print(f'The convolved signal from the module {is_is_not} the same as the np.convolve function.\nThis is because the np.convolve function rounds differently than the function in our module.')

#%% Part 3

#create time vector 
drug_time=np.arange(0,50.01,0.01)
#create input equation
drug_dosage=1-(np.cos(0.25*np.pi*drug_time))

#create system 
h1 = 0.25*np.exp(-drug_time/0.4)*drug_dosage
h2 = 1-np.cos(0.25*np.pi*-drug_time)
h3 = np.exp((-2)*(drug_time-1)**2)
h1_h2_convolve=np.convolve(h1, h2)
body_impulse = np.convolve(h1_h2_convolve, h3)
time_array = np.arange(0, len(body_impulse))*dt

#plot figure 2
plt.figure(3, clear=True, figsize=[8,6])
plt.plot(time_array, body_impulse)
plt.xlabel('Drug Infusion Time (s)')
plt.ylabel('Whole-body Concentration of Drug (A.U.)')
plt.title('Concentration of a Drug in the Human Body Over Time')

#call function in for loop
for denominator in range(2,7,2):
    for amplitude in range(0,4,1):
        drug_dosage_loop = amplitude - (np.cos((1/denominator)*np.pi*drug_time))
        label = f'Denominator Term: {denominator}, Amplitude Term: {amplitude}'
        L2.run_drug_simulations(drug_dosage_loop, body_impulse, dt, label)

plt.legend()
plt.savefig("Drug_Concentration.pdf")
print("To get the maximum drug concentration around a power of 10, the denominator should be 6 and the amplitude term should be 3.")
 

#%% Part 4
#identify file path, read in file
file_path='BassLoop.wav'
audio_data, sample_rate = sf.read(file_path)

#Changing audio data shape to a mono data variable by taking mean across columns
mono_audio_data=np.mean(audio_data, axis=1)

#create clip of song
clip_start = 6*sample_rate
clip_length = 9*sample_rate
song_clip = mono_audio_data[clip_start:clip_length]
sd.play(song_clip, sample_rate)

#create time array
song_time_vector = np.arange(0, len(song_clip))/sample_rate

#create plot of audio
plt.figure(4, clear = True)
plt.plot(song_time_vector, song_clip)
plt.xlabel("Time (s)")
plt.ylabel;("Amplitude (A.U.)")
plt.grid()
plt.title("Song Clip")
plt.savefig("Song_Clip.pdf")

#print statement
print("The plot looks like exactly what you'd think a sound wave looks like!")

#play with doubled sampling rate
doubled_sample_rate = 2*sample_rate
sd.play(song_clip, doubled_sample_rate)
print("We predicted that doubling the sample rate would make the clip sound sped up, and we were correct.")

#play with halved sampling rate
halved_sample_rate = .5*sample_rate
sd.play(song_clip, halved_sample_rate)
print("We predicted that halving the sample rate would make the clip sound slowed down, and we were correct.")

#Convolve audio data with high pass filter
hp_filter_name = "HPF_1000Hz_fs44100_n10001.txt"
hp_filter_array = np.loadtxt(hp_filter_name)
hp_filter_convolved = np.convolve(song_clip, hp_filter_array)
sd.play(hp_filter_convolved, sample_rate) 
print("This made the song sound tinny and low quality.")
save_hp_filter_convolved = "highpass.txt"
np.savetxt(save_hp_filter_convolved, hp_filter_convolved)

#Convolve audio data with low pass filter
lp_filter_name = "LPF_1000Hz_fs44100_n10001.txt"
lp_filter_array = np.loadtxt(lp_filter_name)
lp_filter_convolved = np.convolve(song_clip, lp_filter_array)
sd.play(lp_filter_convolved, sample_rate)
print("This made the song sound muffled, but higher quality than the high pass. ")
save_lp_filter_convolved = "lowpass.txt"
np.savetxt(save_lp_filter_convolved, lp_filter_convolved)

#Design first filter
my_up_filter_array = np.arange(0, 0.02, 0.0004)
my_down_filter_array = np.arange(0.02, -0.0001, -.0004)
my_filter_array = np.concatenate((my_up_filter_array, my_down_filter_array))
my_filter_convolved = np.convolve(song_clip, my_filter_array)
sd.play(my_filter_convolved, sample_rate)
print("This filter also made the song sound muffled, but higher quality than the high pass. ")
save_my_filter_convolved = "my_filter_convolved.txt"
np.savetxt(save_my_filter_convolved, my_filter_convolved)

#Design second filter
h = np.zeros(10002)
h[0] = 1
h[-1] = 1
h_convolved = np.convolve(song_clip, h)
sd.play(h_convolved, sample_rate)
print("This made the sound busy - it sounded like there were multiple instruments at the same time.")
save_h_convolved = "h_convolved.txt"
np.savetxt(save_h_convolved, h_convolved)

#Room as a system that changes sound
#Load room sound file
room_filter_name = 'doorway.wav'
room_audio_data, room_sample_rate = sf.read(room_filter_name)
mono_room_audio = np.mean(room_audio_data, axis=1)

#make sure sample rates match
is_rate_same = np.equal(sample_rate, room_sample_rate)
if is_rate_same == True:
    print("Sample rates are the same.")
else: print("Sample rates are not the same.")

#convolve the song with the room system
convolved_song_room = np.convolve(mono_audio_data, mono_room_audio)
sd.play(convolved_song_room, room_sample_rate)
print("I recorded the impulse response function of the room by recording inside an enclosed double doorway space. I stomped on the ground once during the recording.\n This made the song sound awful! It's too loud, and there are echoes that make the sound quality poor. ")
save_convolved_song_room = "Room_song_convolved.txt"
np.savetxt(save_convolved_song_room, convolved_song_room )
