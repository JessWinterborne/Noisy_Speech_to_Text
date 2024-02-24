#requirements 
#pip install pyanote.audio

from pyannote.audio import Pipeline
import torch

def diarize(audio_file):
    #you will probably need an authorisation token from huggingface the first time you run the model
    #however, you do not need it subsequently, and the model does NOT connect to huggingface every time so it can be run locally!
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="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")
    #send pipeline to GPU (when available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline.to(torch.device(device))

    #apply pretrained pipeline
    diarization = pipeline(audio_file)
    
    speaker_and_time = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_and_time.append([speaker, turn.start])

    for i in range(len(speaker_and_time)-1,0,-1):
        if speaker_and_time[i][0] == speaker_and_time[i-1][0]:
            del speaker_and_time[i]

    # speaker_and_time[0][1] = 0
    return speaker_and_time

#test
# print(diarize('1-Low-ENG-Trimmed_2_converted.wav'))

#TODO test this without internet connection, does it still work?
