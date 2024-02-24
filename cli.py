import argparse
from e2e_pipeline import e2e_pipeline
from translate_nllb import nllb_translate
import os


def main():
    parser = argparse.ArgumentParser(description="Process inputs from command line")
    parser.add_argument("file_name", help="Name of the audio file to process")
    parser.add_argument("noisy", default=None, help="Choose a noise reduction method. Options: dfn if ..., dg if ..., None (default) if ....")
    parser.add_argument("translate_language", nargs='?', default=None, help="Language to translate to as code (i.e. en, nl), leave blank for no translation")
    #TODO fix these inputs 
    parser.add_argument("option", help="Choose diarization/transcribe option 0 or 1 or 2")
    parser.add_argument("--save", action="store_true", help="Would you like to save the results")
    
    args = parser.parse_args()

    file_name = args.file_name
    if args.translate_language:
        translate_language = args.translate_language.lower()
    noisy = args.noisy.lower()
    save = args.save
    if noisy == 'none':
        noisy = None

    if args.option:
        option = args.option
        if option not in ["0","1","2"]:
            print("invalid option")
    else:
        option = "0"

    if noisy not in [None, 'dfn', 'sg']:
        print('Invalid noise reduction option')

    # print(file_name, translate_language, noisy, save)
    print('Running transcription')

    # combined_timestamps, language, full_text = e2e_pipeline(input_file_path=file_name, noise_reduction=noisy, save=save, from_cli=True, option=option)
    result = e2e_pipeline(input_file_path=file_name, noise_reduction=noisy, save=save, from_cli=True, option=option)
    combined_timestamps = result[0]
    language = result[1]
    full_text = result[2]
    print(f'Full transcription: {full_text}')

    if args.translate_language:
        print('Running translation')
        if language == translate_language:
            print(f'Audio is already in language: {translate_language}')
        else:
            translated = nllb_translate(full_text, source_language_input=language, target_language_input=translate_language)
            print(f'Translated transcript: {translated}')

            if save:
                 with open(os.path.join('exports/','translated_text.txt'), mode='w') as txt:
                    txt.write(translated)
    if save:
         print('Files saved in exports/')

    return('Finished !')


if __name__ == "__main__":
    main()

# EXAMPLE USAGE:
# python cli.py 'audio_file.mp3' 'dfn' 'spanish' 'option' --save 
#option relates to whether we run noise reduction or not. please see e2e_pipeline.
