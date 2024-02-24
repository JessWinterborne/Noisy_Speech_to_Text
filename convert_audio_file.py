#requirements:
#make sure ffmpeg is installed on the system:
#sudo apt update && sudo apt upgrade
#sudo apt install ffmpeg

from pydub import AudioSegment

def convert_to_wav(file_name):
    #check that filename is in format 'name.extension', e.g. 'test.wav'
    if not len(file_name.split('.')) == 2:
        print("Please upload a file with only a single . separating the name and file extension")
        #TODO show this message on webpage
        raise ValueError

    file_extension = file_name.split('.')[-1]

    #check the audio file is in a supported format
    if file_extension.lower() not in ['mp3', 'wav', 'ogg', 'mp4', 'flv', 'wma', 'aac', 'm4a']:
        #TODO check .flac and add it on
        print('File type is not supported, please upload a wav, mp3, ogg, mp4, flv, wma, m4a or aac file')
        #TODO show this message on webpage
        raise ValueError

    print(f'Converting {file_extension} file: {file_name} to .wav')

    #using a 16kHz sample rate as this is what Whisper takes
    audio_to_convert = AudioSegment.from_file(file_name, file_extension.lower())

    audio_to_convert = audio_to_convert.split_to_mono()[0]

    #adding _converted to the file name
    new_file_name = '{}_converted.wav'.format(file_name.split('.')[0])

    #exporting new .wav file
    audio_to_convert.export(new_file_name, format='wav', parameters=["-ar", "16000"])

    #returning new file name for use in e2e pipeline function
    return new_file_name

# # test
print(convert_to_wav('Background_Chatter_Medium.mp3'))