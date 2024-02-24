#requirements
#will need ffmpeg installed on the system:
#sudo apt update && sudo apt upgrade
#sudo apt install ffmpeg

#TODO make read me and put extra requirements in it (non pip)
#check about pip3 and whether this needs to be in there: pip3 install torch torchvision torchaudio 
#note as well this is only for mac.

import whisper
import torch

#function to transcribe audio 
def whisper_transcribe(file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #loading the whisper model from the models folder
    stt_model = whisper.load_model('models/base.pt', device=device)
    #runs the model and saves the output (type = dictionary)
    stt_model_output = stt_model.transcribe(file_path, word_timestamps = True)
    full_text = stt_model_output['text']
    language = stt_model_output['language']

    word_and_timestamp = []
    for segment in stt_model_output['segments']:
        for item in segment['words']:
            word_and_timestamp.append((item['word'], item['start']))
    
    return full_text, language, word_and_timestamp

# print(whisper_transcribe('white_noise_dfn.wav'))