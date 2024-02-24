import noisereduce as nr
from scipy.io import wavfile
import numpy as np

def noise_reduction_sg(input_file_name):
    print('Running noise reduction using spectral gating')

    #this saves the .wav file as the audio data, and the sample rate
    samplerate, data = wavfile.read(input_file_name)
    #data could be 1d for 1 channel wav or 2d for 2 channel wav, with the shape = (no. samples, no. channels)
    #in the data is the the amplitudes for each sample I believe (for each channel)

    # if np.shape(data)[1] > 1:
    #     data = data[:,0]

    #saving the noise reduced audio data and sample rate 
    output_audio_bytes = nr.reduce_noise(y=data, sr=samplerate, stationary=True)
    print('Noise reduction complete')

    #new noise reduced file name with _DFN added onto the end
    sg_file_name = '{}_sg.{}'.format(input_file_name.split('.')[0],input_file_name.split('.')[-1])

    #writing the new .wav file
    wavfile.write(sg_file_name, 16000, output_audio_bytes)
    #sets the sample rate at 16,000 (this is what whisper takes in anyway)

    #returning new file name for use in e2e pipeline function
    return sg_file_name

# #test
# print(noise_reduction_sg('1-Low-ENG-Trimmed_2_converted.wav'))


