#we need to import all of our functions for the pipeline:
from convert_audio_file import convert_to_wav
from noise_reduction_dfn import noise_reduction_dfn
from whisper_transcribe import whisper_transcribe
from noise_reduction_sg import noise_reduction_sg
from diarization import diarize
from combine_timestamps import combine_timestamp_data
import os
import json
import csv

#pipeline function
def e2e_pipeline(input_file_path, noise_reduction=None, save=False, from_cli=False, option='0'):
    """
    noise_reduction options: 'dfn', 'sg', None (default)
    order options: option: 0 - normal, 1 - diarise noise reduced, whisper the original, 2 - diarise original, whisper the noise reduced
    """
    
    converted_file_name = convert_to_wav(input_file_path)
    global noise_reduction_file_name

    #now we do the noise reduction
    if noise_reduction == 'dfn':
        noise_reduction_file_name = noise_reduction_dfn(converted_file_name)
    elif noise_reduction == 'sg':
        noise_reduction_file_name = noise_reduction_sg(converted_file_name)
    elif noise_reduction is None: 
        noise_reduction_file_name = converted_file_name
    else:
        print('Invalid noise reduction input')

    if option == "1":
        audio_for_diarization = noise_reduction_file_name
        audio_for_whisper = converted_file_name

    elif option == "2":
        audio_for_whisper = noise_reduction_file_name
        audio_for_diarization = converted_file_name

    else:
        audio_for_diarization = noise_reduction_file_name
        audio_for_whisper = noise_reduction_file_name

    #run diarisation 
    speakers_and_times = diarize(audio_file=audio_for_diarization)

    #run whisper
    # print(whisper_transcribe(file_path=noise_reduction_file_name))
    result = whisper_transcribe(file_path=audio_for_whisper)
    full_text = result[0]
    language = result[1]
    word_and_timestamp = result[2]
    
    combined_timestamps = combine_timestamp_data(speakers_and_times, word_and_timestamp)

    if from_cli:
        os.makedirs('exports', exist_ok = True)
        save_folder = 'exports/'
    else:
        save_folder = 'static/downloads/'

    new_combined_timestamps = []
    for speakers_and_words in combined_timestamps:
        if len(speakers_and_words[1]) != 0:
            new_combined_timestamps.append(speakers_and_words)

    combined_timestamps = new_combined_timestamps

    global start_time
    start_time = 0

    if save:
        data_for_csv = []
        for speaker_and_words in combined_timestamps:
            # print(speaker_and_words)
            if len(speaker_and_words)>1:
                if len(speaker_and_words[1])>0:
                    if len(speaker_and_words[1][0])>1:
                        start_time = speaker_and_words[1][0][1]
            current_speaker_words = ''
            for word_and_timestamp_iterate in speaker_and_words[1]:
                # print(word_and_timestamp_iterate)
                current_speaker_words += word_and_timestamp_iterate[0]
            data_for_csv.append([speaker_and_words[0][0], current_speaker_words, start_time])

        with open(os.path.join(save_folder,'output.csv'), mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["speaker", "text", "time"])
            for row in data_for_csv:
                csv_writer.writerow(row)

        with open(os.path.join(save_folder,'diarized_and_timestamps.json') ,mode='w') as json_file:
            json.dump(combined_timestamps, json_file)


        with open(os.path.join(save_folder,'only_text.txt'), mode='w') as txt:
            txt.write(full_text)


    return combined_timestamps, language, full_text 

# # test
# print(e2e_pipeline('1-Low-ENG-Trimmed_2.mp3', noise_reduction=None, save=True))