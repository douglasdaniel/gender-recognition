#!/usr/bin/enyv python

# file: audacious.py

# Author: Daniel Douglas 

# Function imports
#
import math
import copy
from scipy.io import wavfile
import subprocess
import numpy as np
import wave
import matplotlib.pyplot as plt
import pylab

# Class: Speech
#
# arguments:
#  infile (audio file)
#
# Class used to perform various speech signal processing
# operations. 
#
class Speech(object):



#--------------------------------------------------------    
    # function: __init__
    #
    # arguments:
    #   infile (audio file) audio file to be processed
    #
    # instance attributes:
    #  name (string)
    #  fs (int)
    #  data (numpy.ndarray)
    #  useful_data (numpy.ndarray)
    #  length (int)
    #  channels (int)
    #  signal (numpy.ndarray)
    #
    # Class constructor
    #
#--------------------------------------------------------
    def __init__(self, infile):
        
        # Name of input file
        #
        self.name = infile

        # WAV file label
        #
        if infile.endswith('.wav'):

            self.wavname = infile

        else:

            # Output of MP3 to WAV conversion
            #
            self.wavname = self.mp32wav()
        
        # Sample rate and audio data
        #
        (self.fs, self.data, self.useful_data) = self.danscan1()

        # Length of audio signal in samples
        #
        self.length = len(self.data)

        # Number of channels in audio signal
        #
        self.channels = len(self.data.shape)

        # Audio data with zeros trimmed
        #
        self.signal = self.trimzeros()

    # end function
    #


#---------------------------------------------------------
    # function: mp32wav
    #
    # arguments: 
    #  self
    #
    # return: wavname (string) name of the output WAV file
    #
    # Converts an MP3 file to WAV
    # Places new file directory specified by 'wavname'
    #
#---------------------------------------------------------
    def mp32wav(self):

        # Define label for the output WAV file
        #
        wavname = self.name + '.wav'

        # Open mp3 and convert to WAV
        #
        try:
            subprocess.check_call(['mpg123', '-qw', wavname, self.name])
        except:
            print "File Open Failed. MP3 filename was not valid."
            exit()

        # Return label for the new WAV file (string)
        #
        return wavname

    # end function
    #


#---------------------------------------------------------
    # function: danscan1
    #
    # arguments:
    #  self
    #
    # return: 
    #  fs (int) sample rate in Hz
    #  data (numpy.ndarray) audio data of input file
    #
    # Determines the sample rate of the input file in
    # samples/second
    #
#---------------------------------------------------------
    def danscan1(self):

        # Read the file and extract the sample rate and data
        # Audio data is stored in a NumPy ndarray.
        # Audio channels are represented by array columns.
        #
        (samprate, data) = wavfile.read(self.wavname)
        
        # Convert data array in to normal float32 values
        # in range -1 to 1
        #
        data_norm = self.pcm2float(data, 'float32')
        
        # Remove blank channels from the signal
        #
        data_norm_use = self.chanscan(data_norm)
        
        # Return sample rate and audio data as tuple
        #
        return (float(samprate), data_norm, data_norm_use)

    # end function
    #


#---------------------------------------------------------
    # function: chanscan
    #
    # arguments:
    #  data_signal (numpy.ndarray) Array of N dimensions
    #
    # return: chans (tuple) the non blank columns of array
    #
    # Reads the coulumns of an numpy array and returns the
    # non empty ones
    #
#---------------------------------------------------------
    def chanscan(self, data_signal):

        chans = ()
        
        # Determine the active channels in signal
        # 
        num_channel = len(data_signal.shape)

        if (num_channel > 1):

            # Determine useful channels in signal
            #
            for i in range(num_channel):
                
                # If the channel has any useful data
                #
                if (sum( data_signal[:,i] ) !=  0):

                    # We will keep it
                    #
                    chans = chans + (i,)

            # Return a tuple of the useful channels
            #
            return data_signal[:,chans]

        else:
            return data_signal
        
    # end function
    #


#---------------------------------------------------------
    # function: description
    #
    # arguments:
    #  self 
    #
    # return: none
    #
    # Print out descriptive data about the input audio file
    #
#---------------------------------------------------------
    def description(self):
        
        # Print the sample rate, number of time samples, and
        # number of channels in human readable way.
        #
        print "File : {} : sampling rate = {} Hz, length = {} samples, channels = {}"\
            .format(self.name, self.fs, self.length, self.channels)

    # end function
    #


#---------------------------------------------------------
    # function: pcm2float
    #
    # arguments:
    #  sig: (array_like) Input array, must have integral type.
    #  dtype: (data type), optional, Desired (floating point) data type.
    #
    # returns: 
    #  (numpy.ndarray) Normalized floating point data.
    #
    # Convert PCM signal to floating point with a range from -1 to 1.
    # Use dtype='float32' for single precision.
    #
#---------------------------------------------------------
    def pcm2float(self, sig, dtype='float64'):
        sig = np.asarray(sig)
        if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max
    
    # end function
    #


#---------------------------------------------------------
    # function: trimzeros
    #
    # arguments:
    #  self
    #  
    # return:
    #  (numpy ndarray) input file audio data with leading and
    #   trailing zeros removed
    #
    # Removes the leading and trailing zeros from a signal
    #
#----------------------------------------------------------
    def trimzeros(self):

        index = []

        # Sum the two channels of each time enrty.
        # Remove time entry of leading zeros
        #
        for i in range(len(self.useful_data)):

            if (self.channels > 1):
                tmp = sum(self.useful_data[i])
            else:
                tmp = self.useful_data[i]

            if ( tmp  < 0.0005 ):
                index.append(i)
            else:
                break
        
        # Sum the two channels of each entry.
        # Remove time entry of trailing zeros.
        #
        for j in range(len(self.useful_data)-1, 0, -1):
            
            if (self.channels > 1):
                tmp = sum(self.useful_data[j])
            else:
                tmp = self.useful_data[j]



            if ( tmp  < 0.0005 ):
                index.append(j)
            else:
                break
            
        # Create a copy of the audio data with the leading
        # and trailing zeros removed.
        #
        data_trimm = np.delete(self.useful_data, index, 0)

        # Return zero trimmed array
        #
        return data_trimm

    # end function
    #


#---------------------------------------------------------
    # function: hamming
    #
    # arguments:
    #  self
    #  size (int) size of window in samples
    #
    # return:
    #  win (numpy.array) hamming window coefficients
    #
    # Create a Hamming window of coefficients distributed
    # over the range [0,1]
    #
#---------------------------------------------------------
    def hamming(self, size):

        # Initialize empty array
        #
        win = np.zeros(size)

        # Fill array with Hamming coefficients
        #
        for i in range(size):
            
            # Scale factor
            #
            xr = float(i) / float(size)

            # Compute coefficient using Hamming window function
            #
            tmp = 0.54 - 0.46 * math.cos( 2.0 * math.pi * xr )

            # Add coefficient to window
            #
            win[i] = tmp

        # Return the window
        #
        return win
        
    # end function
    #


#---------------------------------------------------------
    # function: findmax 
    # 
    # arguments:
    #  data a numpy array
    #  offsetLeft the index position at which analysis will commence
    #  offsetRight the terminating index position. if -1, the array size 
    #   will be used
    #
    # return: a list containing the index and the value of the maximum
    #
    # Find the maximum value and index in an array
    #
#---------------------------------------------------------
    def findmax(self,
		data, 
		offsetLeft = 0, 
		offsetRight = -1, # if -1, the array size will be used
    ):
	
        objType = type(data).__name__.strip()
	if objType <> "ndarray":
		raise Exception('data argument is no instance of numpy.array')
	size = len(data)
	if (size < 1):
		raise Exception('data array is empty')
	xOfMax = -1
	valMax = min(data)
	if offsetRight == -1:
		offsetRight = size
	for i in range(offsetLeft + 1, offsetRight - 1):
		if data[i] >= data[i-1] and data[i] >= data[i + 1]:
			if data[i] > valMax:
				valMax = data[i]
				xOfMax = i

        return [xOfMax, valMax]
    
    # end function
    #

    
#---------------------------------------------------------
    # function: findF0once
    #
    # arguments:
    #  data (numpy.array) array or a list of float
    #  fs (int) signal sampling frequency in Hz
    #  Fmin (int) lowest possible F0 Hz
    #  Fmax (int) highest possible F0 in Hz
    #  voice_threshold (float) threshold of AC maximum
    #
    # return: Estimated fundamental frequency in Hz
    #  
    # Calculates the fundamental frequency of a given
    # signal. Treats signal as one stationary block
    # and calculates one F0.
    #
#---------------------------------------------------------
    def findF0once(self,
        data,
        fs,
        Fmin = 50,
        Fmax = 3000,
        voicingThreshold = 0.3,
    ):

	data_tmp = data

	# Apply Hamming window function
        #
        window = self.hamming(len(data_tmp))
        data_tmp = data_tmp.reshape(len(data_tmp),)
       
        data_tmp *= window 
	
	# Take autocorrelation of signal
        #
	result = np.correlate(data_tmp, data_tmp, mode = 'full')
	r = result[result.size/2:] / float(len(data))
        
	# find peak energy in autocorrelation
        #
        xOfMax, valMax = self.findmax(r, fs / Fmax, fs / Fmin)
        valMax /= max(r)
        freq = fs / xOfMax
	return freq

    # end function
    #


#----------------------------------------------------------
    # function: findceps
    #
    # arguments:
    #  self
    #  data (numpy array) signal data
    #  n (int) size of the FFT
    #
    # return: ceps (numpy array) cepstrum
    #
    # Compute the complex cepstrum of signal data
    #
#----------------------------------------------------------
    def findceps(self, data, n=None):
        
        # Compute the phase offset to add to the cepstrum
        #
        '''def _unwrap(phase):
            samples = phase.shape[-1]
            unwrapped = np.unwrap(phase)
            center = (samples+1)//2
            if samples == 1: 
                center = 0  
            ndelay = np.array(np.round(unwrapped[...,center]/np.pi))
            unwrapped -= np.pi * ndelay[...,None] * np.arange(samples) / center
            return unwrapped, ndelay'''
        
        # Compute the real magnitude spectrum
        #
        win = self.hamming(len(data))

        data_win = data * win

        spectrum = np.fft.fft(data_win, n=n)

        # Compute the phase offset
        #
        #unwrapped_phase, ndelay = _unwrap(np.angle(spectrum))
        
        # Take the logarithm of the spectrum
        #
        log_spectrum = np.log(np.abs(spectrum))
        
        # Compute the cepstrum
        #
        ceps = np.fft.ifft(log_spectrum).real

        return ceps

    # end function
    #


#---------------------------------------------------------
    # function: frames
    # 
    # arguments:
    #  self
    #  msec (int) number of milliseconds per frame
    #  overlap (int) number of milliseconds of frame
    #   overlap
    #  
    # return: signal_framed (list of numpy.ndarray)
    #
    # Break the signal up into overlapping frames
    #
#----------------------------------------------------------
    def frames(self, msec, overlap, data):

        # Length of frame in samples
        #
        frlen = int(msec * (self.fs / 1000))

        # Length of overlap in samples
        #
        ovlen = int(overlap * (self.fs/1000))

        # Split signal
        #
        framed = []
        step = xrange(0,len(data),frlen - ovlen)

        for i in step:
            y = data[i:i+frlen]
            y = y.reshape(y.size,)

            if len(y) < frlen:
                # Pad zeros on last frame if length is short
                #
                extension = frlen - len(y)
                np.pad(y, (0,extension), 'constant')
            framed.append(y)
        
        return framed


#--------------------------------------------------------
    # function: findF0
    #
    # arguments:
    #
    # return: pitch (list of float) a list of F0 for each 
    #  frame.
    #
    # Compute the fundamental frequency using the real
    # cepstrum
    #
#---------------------------------------------------------
    def findF0(self, flen, ovlp):

        frames = self.frames(flen, ovlp, self.signal)

        pitch = []

        for frame in frames:

            # Compute the complex cepstrum for the frame
            #
            cepstrum = self.findceps(frame)

            # Apply some liftering
            # 
            cepstrum *= self.hamming(len(cepstrum))
            
            # Determine time axis
            #
            t = [x / self.fs for x in range(0,len(frame))]
            tarr = np.asarray(t)
            tarr = np.reshape(tarr, (len(tarr), 1))
            
            # Concatenate the cepstral and quefrency axes
            #
            cepstrumc = cepstrum[:,None]
            cepfrency = np.append(cepstrumc, tarr, 1)
            
            # Find desired time (quefrency) range
            #
            start = 0.0028 # seconds (350 Hz)
            stop = 0.0125 # seconds (80 Hz)
            for i in range(0,len(tarr)):
                if tarr[i] <= start:
                    istart = i
                if tarr[i] >= stop:
                    istop = i
                    break

            # Search cepstrum range for maximum
            #
            peak = max(cepfrency[istart:istop+1,0])
            ipeak = np.where(cepfrency[:,0] == peak)
            ipeak = [x[0] for x in ipeak][0]

            # Quefrency at peak
            #
            quefrency = cepfrency[ipeak,1]

            # Frequency
            #
            freq = math.floor(100 / quefrency) / 100
            
            # Store pitch for this frame
            #
            pitch.append(freq)

        #
        return pitch
                        
    # end function
    #


#---------------------------------------------------------
    # function: gender
    #
    # arguments:
    #  self
    #  flen (int) frame length in milliseconds
    #  ovlp (int) window overlap (milliseconds)
    #
    # return: gender (str), likely (str)
    #
    # Use the extimated values of F0 to determine the
    # gender of the speaker
    #
#---------------------------------------------------------
    def gender(self, flen=30, ovlp=3):

        pitches = self.findF0(flen, ovlp)

        # Focus data around adult speech
        #
        copy = [f for f in pitches if f > 110.0 and f < 260.0]
        
        min_f0 =  min(copy)
        max_f0 =  max(copy)

        average_f0 = math.floor( 100 * sum(copy) / len(copy) ) / 100

        # Counters
        #
        M = 0
        F = 0

        # For each frame, take the value of F0 and guess
        # the gender of speaker
        #
        for frq in copy:
            if frq < 160.0:
                M += 1
            else:
                F += 1

        # Summarize findings
        #
        if M > F:
            conc= 'GENDER IS MALE'
        elif M < F:
            conc= 'GENDER IS FEMALE'
        else:
            conc= 'GENDER INDETERMINABLE'

        return conc
        
