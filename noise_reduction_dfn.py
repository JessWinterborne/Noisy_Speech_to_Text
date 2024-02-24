#command line requirements:
#pip install deepfilternet
#pip install numpy
#pip install scipy
#pip3 install torch torchvision torchaudio (need cuda 12.1 - see pytorch.org/get-started/locally/)

#put DeepFilterNet3 folder into models folder

from pathlib import Path
from df.enhance import enhance, init_df
import numpy as np
import torch
from typing import Union
from scipy.io import wavfile

######### audio_tools methods from https://github.com/Sharrnah/whispering #########
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

def _resample(smp, scale=1.0):
    """Resample a sound to be a different length

    Sample must be mono.  May take some time for longer sounds
    sampled at 44100 Hz.

    Keyword arguments:
    scale - scale factor for length of sound (2.0 means double length)

    """
    # f*ing cool, numpy can do this with one command
    # calculate new length of sample
    n = round(len(smp) * scale)
    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    return np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(smp), endpoint=False),  # known positions
        smp,  # known data points
    )

def _interleave(left, right):
    """Given two separate arrays, return a new interleaved array

    This function is useful for converting separate left/right audio
    streams into one stereo audio stream.  Input arrays and returned
    array are Numpy arrays.

    See also: uninterleave()

    """
    return np.ravel(np.vstack((left, right)), order='F')

def _uninterleave(data):
    """Given a stereo array, return separate left and right streams

    This function converts one array representing interleaved left and
    right audio streams into separate left and right arrays.  The return
    value is a list of length two.  Input array and output arrays are all
    Numpy arrays.

    See also: interleave()

    """
    return data.reshape(2, len(data) // 2, order='F')

def resample_audio(audio_chunk, recorded_sample_rate, target_sample_rate, target_channels=-1, is_mono=None,
                   dtype="int16"):
    """
    Resample audio data and optionally convert between stereo and mono.

    :param audio_chunk: The raw audio data chunk as bytes, NumPy array or PyTorch Tensor.
    :param recorded_sample_rate: The sample rate of the input audio.
    :param target_sample_rate: The desired target sample rate for the output.
    :param target_channels: The desired number of channels in the output.
        - '-1': Average the left and right channels to create mono audio. (default)
        - '0': Extract the first channel (left channel) data.
        - '1': Extract the second channel (right channel) data.
        - '2': Keep stereo channels (or copy the mono channel to both channels if is_mono is True).
    :param is_mono: Specify whether the input audio is mono. If None, it will be determined from the shape of the audio data.
    :param dtype: The desired data type of the output audio, either "int16" or "float32".
    :return: A NumPy array containing the resampled audio data.
    """
    # Determine the data type for audio data
    audio_data_dtype = np.float32
    if dtype == "int8":
        audio_data_dtype = np.int8
    elif dtype == "int16":
        audio_data_dtype = np.int16
    elif dtype == "int32":
        audio_data_dtype = np.int32
    elif dtype == "float32":
        audio_data_dtype = np.float32

    # Convert the audio chunk to a numpy array
    if isinstance(audio_chunk, torch.Tensor):
        audio_chunk = audio_chunk.detach().cpu().numpy()

    audio_data = np.frombuffer(audio_chunk, dtype=audio_data_dtype)

    # Determine if the audio is mono or stereo; assume mono if the shape has one dimension
    if is_mono is None:
        is_mono = len(audio_data.shape) == 1

    # If stereo, reshape the data to have two columns (left and right channels)
    if not is_mono:
        audio_data = audio_data.reshape(-1, 2)

    # Handle channel conversion based on the target_channels parameter
    # -1 means converting stereo to mono by taking the mean of both channels
    # 0 or 1 means selecting one of the stereo channels
    # 2 means duplicating the mono channel to make it stereo
    if target_channels == -1 and not is_mono:
        audio_data = audio_data.mean(axis=1)
    elif target_channels in [0, 1] and not is_mono:
        audio_data = audio_data[:, target_channels]
    elif target_channels == 2 and is_mono:
        audio_data = _interleave(audio_data, audio_data)

    # Calculate the scaling factor for resampling
    scale = target_sample_rate / recorded_sample_rate

    # Perform resampling based on whether the audio is mono or stereo
    # If mono or selected one channel, use _resample directly
    # If stereo, split into left and right, resample separately, then interleave
    if is_mono or target_channels in [0, 1, -1]:
        audio_data = _resample(audio_data, scale)
    else:  # Stereo
        left, right = _uninterleave(audio_data)
        left_resampled = _resample(left, scale)
        right_resampled = _resample(right, scale)
        audio_data = _interleave(left_resampled, right_resampled)

    # Return the resampled audio data with the specified dtype
    return np.asarray(audio_data, dtype=audio_data_dtype)

######### DeepFilterNet from https://github.com/Sharrnah/whispering #########
#a class which lets you input audio data and sample rate, and applys noise reduction (from deep filter net 3 model in models folder) 
#and outputs data and sample rate
class DeepFilterNet(metaclass=SingletonMeta):
    df_model = None
    df_state = None

    def __init__(self, post_filter=False, epoch: Union[str, int, None] = "best"):
        # os.makedirs(cache_df_path, exist_ok=True)

        model = "models/DeepFilterNet3"

        self.df_model, self.df_state, _ = init_df(model_base_dir=str(Path(model).resolve()), post_filter=post_filter, epoch=epoch, log_level="none")

        pass

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def enhance_audio(self, audio_bytes, sample_rate=16000, output_sample_rate=16000, is_mono=True):
        enhanced_sample_rate = self.df_state.sr()
        audio_bytes = resample_audio(audio_bytes, sample_rate, enhanced_sample_rate, -1,
                                                 is_mono=is_mono).tobytes()

        audio_full_int16 = np.frombuffer(audio_bytes, np.int16)
        audio_bytes = self.int2float(audio_full_int16)

        audio_tensor = torch.frombuffer(audio_bytes, dtype=torch.float32).unsqueeze_(0)
        # convert bytes to torch tensor
        enhanced_audio = enhance(self.df_model, self.df_state, torch.as_tensor(audio_tensor))
        # convert torch tensor to bytes
        enhanced_audio = torch.as_tensor(enhanced_audio)

        if enhanced_audio.ndim == 1:
            enhanced_audio.unsqueeze_(0)

        if enhanced_audio.dtype != torch.int16:
            enhanced_audio = (enhanced_audio * (1 << 15)).to(torch.int16)
        elif enhanced_audio.dtype != torch.float32:
            enhanced_audio = enhanced_audio.to(torch.float32) / (1 << 15)

        enhanced_audio = enhanced_audio.squeeze().numpy().astype(np.int16).tobytes()

        audio_bytes = resample_audio(enhanced_audio, enhanced_sample_rate, output_sample_rate, -1,
                                                 is_mono=True)

        # clear variables
        enhanced_audio = None
        del enhanced_audio
        audio_tensor = None
        del audio_tensor
        audio_full_int16 = None
        del audio_full_int16

        return audio_bytes


######### Our Main Function ######### 
#our function that uses the deep filter net class above to do noise reduction on our inputted .wav audio file
#outputs a .wav file again
def noise_reduction_dfn(input_file_name):
    print('Running noise reduction using DeepFilterNet')

    #saving the class above 
    dfn = DeepFilterNet()

    #this saves the .wav file as the audio data, and the sample rate
    samplerate, data = wavfile.read(input_file_name)
    #data could be 1d for 1 channel wav or 2d for 2 channel wav, with the shape = (no. samples, no. channels)
    #in the data is the the amplitudes for each sample I believe (for each channel)

    # if np.shape(data)[1] > 1:
    #     data = data[:,0]
    #commented out as changed the audio to mono always

    #saving the noise reduced audio data and sample rate 
    output_audio_bytes = dfn.enhance_audio(data, sample_rate=samplerate)
    # print('Noise reduction complete')

    #new noise reduced file name with _DFN added onto the end
    dfn_file_name = '{}_dfn.{}'.format(input_file_name.split('.')[0],input_file_name.split('.')[-1])

    #writing the new .wav file
    wavfile.write(dfn_file_name, 16000, output_audio_bytes)
    #sets the sample rate at 16,000 (this is what whisper takes in anyway)

    #returning new file name for use in e2e pipeline function
    return dfn_file_name

# #test
print(noise_reduction_dfn('Background_Chatter_Medium_converted.wav'))