#pip install transformers
#install model

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# def nllb_translate(input_text, source_langage='eng_Latn', target_language='nld_Latn'):
def nllb_translate(input_text, source_language_input, target_language_input):

    # print(source_language_input, target_language_input)
    global source_language
    global target_language
    # convert languages
    # whisper uses text or ISO 639 set 1 codes
    # NLLB uses FLORES-200 codes
    if source_language_input.lower() == 'english':
        source_language = 'eng_Latn'
    elif source_language_input.lower() == 'nl' or target_language_input.lower() == 'dutch':
        source_language = 'nld_Latn'
    elif source_language_input.lower() == 'de' or target_language_input.lower() == 'german':
        source_language = 'deu_Latn'
    elif source_language_input.lower() == 'fr' or target_language_input.lower() == 'french':
        source_language = 'fra_Latn'
    elif source_language_input.lower() == 'es' or target_language_input.lower() == 'spanish':
        source_language = 'spa_Latn'
    elif source_language_input.lower() == 'ru' or target_language_input.lower() == 'russian':
        source_language = 'rus_Cyrl'
    # elif source_language_input.lower() == 'cy' or source_language_input.lower() == 'welsh':
    #     source_language = 'cym_Latn'
    elif source_language_input.lower() == 'zh' or target_language_input.lower() == 'chinese':
        #chinese simplified
        source_language = 'zho_Hans'
    else:
        print('Not supported language')

    if target_language_input.lower() == 'english':
        target_language = 'eng_Latn'
    elif target_language_input.lower() == 'nl' or target_language_input.lower() == 'dutch':
        target_language = 'nld_Latn'
    elif target_language_input.lower() == 'de' or target_language_input.lower() == 'german':
        target_language = 'deu_Latn'
    elif target_language_input.lower() == 'fr' or target_language_input.lower() == 'french':
        target_language = 'fra_Latn'
    elif target_language_input.lower() == 'es' or target_language_input.lower() == 'spanish':
        target_language = 'spa_Latn'
    elif target_language_input.lower() == 'ru' or target_language_input.lower() == 'russian':
        target_language = 'rus_Cyrl'
    # elif target_language_input.lower() == 'cy' or target_language_input.lower() == 'welsh':
    #     target_language = 'cym_Latn'
    elif target_language_input.lower() == 'zh' or target_language_input.lower() == 'chinese':
        #chinese simplified
        target_language = 'zho_Hans'
    else:
        print('Not supported language')

    max_chunk_length = 400

    def break_string_into_chunks(input_string, max_chunk_length=max_chunk_length):
        words = input_string.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(' '.join(current_chunk + [word])) <= max_chunk_length:
                current_chunk.append(word)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


    # print(f'TRANSLATE func input {input_text}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 0 if torch.cuda.is_available() else -1

    # Load the model from the model folder
    model = AutoModelForSeq2SeqLM.from_pretrained('models/nllb-200-distilled-600M')
    tokenizer = AutoTokenizer.from_pretrained('models/nllb-200-distilled-600M')

    translation_pipeline = pipeline('translation',
                                        model=model,
                                        tokenizer=tokenizer,
                                        src_lang=source_language,
                                        tgt_lang=target_language,
                                        max_length=400,
                                        device=device)
    
    input_text = input_text.replace('.','')
    input_text = input_text.lower()
    #hacky fix
    input_text = 'Hello world.' + input_text
    
    if len(input_text) >= max_chunk_length:
        chunks = break_string_into_chunks(input_text)
        translated_text = ''
        for chunk in chunks:
            result = translation_pipeline(chunk)
            translated_chunk = result[0]['translation_text']

            translated_text += str(translated_chunk)+' '
    else:
        # print("no chunks")
        result = translation_pipeline(input_text)
        translated_text = result[0]['translation_text']

    #hacky solution to get it to translate the first sentence
    translated_text = translated_text.split('.')[-1]
    # print(f'TRANSLATE func output {translated_text}')
    return(translated_text)

# test = 'Plan are well underway for races to Mars and the Moon in 1992 by Solar Sale. The race to Mars is to commemorate Columbuss journey to the new world 500 years ago. and the one to the Moon is to promote the use of solar sales in space exploration.'
# print(nllb_translate(test, source_language_input='english', target_language_input='welsh'))
