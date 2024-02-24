import werpy
from e2e_pipeline import e2e_pipeline
import string

#function calculating the word error rate 
def word_error_rate(transcribed_text, true_text):
    wer = werpy.wer(transcribed_text, true_text)
    return 100*wer

#transcripting the files
def error_rate(file_path, true_text):
    transcript, language, word_and_timestamp = e2e_pipeline(file_path, noise_reduction='sg')
    transcript = transcript.translate(str.maketrans('', '', string.punctuation))
    with open(true_text, 'r') as file:
        true_transcript = file.read()
        true_transcript = true_transcript.translate(str.maketrans('', '', string.punctuation))
        wer = word_error_rate(transcript.lower(), true_transcript.lower())
    return wer


# test
# print(error_rate('white_noise.wav', 'true_text_white_noise.txt'))







