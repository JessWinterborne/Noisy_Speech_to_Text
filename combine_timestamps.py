# # dia_result = [('SPEAKER_00', 0.01), ('SPEAKER_00', 4.76), ('SPEAKER_01', 7.02), ('SPEAKER_01', 11.62), ('SPEAKER_00', 13.62)]
# speakers_and_times = [('SPEAKER_00', 0.01), ('SPEAKER_01', 7.02), ('SPEAKER_00', 13.62)]

# words_and_times = [(' Plan', 0.0), (' the', 0.46), (' well', 0.68), (' underway', 0.92), (' for', 1.32), (' races', 1.76), (' to', 2.06), (' Mars', 2.44), (' and', 2.64), (' the', 2.8), (' Moon', 2.92), (' in', 3.08), (' 1992', 3.42), (' by', 4.2), (' Solar', 5.08), (' Sale.', 5.42), (' The', 7.04), (' race', 7.18), (' to', 7.38), (' Mars', 7.54), (' is', 7.82), (' to', 8.08), (' commemorate', 8.18), (" Columbus's", 8.6), (' journey', 9.1), (' to', 9.34), (' the', 9.58), (' new', 9.64), (' world', 9.86), (' 500', 10.18), (' years', 10.66), (' ago,', 10.96), (' and', 11.56), (' the', 11.82), (' one', 11.88), (' to', 12.02), (' the', 12.16), (' Moon', 12.26), (' is', 12.64), (' to', 12.98), (' promote', 13.06), (' the', 13.3), (' use', 13.46), (' of', 13.66), (' solar', 13.94), (' sales', 14.14), (' in', 14.64), (' space', 15.16), (' exploration.', 15.44)]

def combine_timestamp_data(speakers_and_times_data, words_and_times_data):
    """
    speakers_and_times: output of diarization.diazire()
    word_and_timestamp: output of whisper_transcribe.whisper_transcribe()
    """

    speakers_and_times_data[0][1] = words_and_times_data[0][1]
    result_list = []
    for i in range(len(speakers_and_times_data)): 
        speaker, start_time = speakers_and_times_data[i]
        next_start_time = speakers_and_times_data[i+1][1] if i+1 < len(speakers_and_times_data) else float('inf')

        speaker_data = [speaker]
        words_data = []

        for word, word_time in words_and_times_data:
            if start_time <= word_time < next_start_time:
                words_data.append((word,word_time))
        
        result_list.append([speaker_data, words_data])
    return result_list

# print(result_list)
