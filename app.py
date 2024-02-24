from flask import Flask, render_template, send_file, request, jsonify
import shutil
import os
from werkzeug.utils import secure_filename
from translate_nllb import nllb_translate
import sys
import torch

from e2e_pipeline import e2e_pipeline

app = Flask(__name__)

global full_text
global source_language

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'mp4', 'flv', 'wma', 'aac', 'm4a'}
#setting the name of the uploads folder (??)
uploads_folder = 'uploads'
#telling flask where the uplaod folder is (??)
app.config['UPLOAD_FOLDER'] = uploads_folder
#making the uploads folder
os.makedirs(uploads_folder, exist_ok = True)
#making the folder to store the converted audio file in for later playback
os.makedirs('static/audio/', exist_ok = True)

#TODO: this is from flask tutorial but can probably re-write nicer
#put file name in function and tells us if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#function to delete everything in a given folder.
def clear_folder(folder_name):
    for root, dirs, files in os.walk(folder_name):
        for f in files:
            #have to unlink files in the folder first (??)
            os.unlink(os.path.join(root, f))
        for d in dirs:
            #rmtree = remove tree, this deletes all the files in the folder
            shutil.rmtree(os.path.join(root, d))

#function copys the converted .wav file to the static/audio folder
def copy_wav_file(original_filename):
        #define the source and destination file paths
        source_path = os.path.join('uploads/', str(original_filename))
        #always has the same name 'playback.wav' in the static audio folder
        destination_path = os.path.join('static/audio/', 'playback.wav')

        #check if the source file exists
        if os.path.exists(source_path):
            #copy the file from source to destination
            shutil.copy(source_path, destination_path)
            print(f"File copied successfully from {source_path} to {destination_path}")
        else:
            print(f"Source file {source_path} not found. Unable to copy.")
    
#running the clear uploads folder on startup in case theres anyhing left in it
#in case of crashes 
clear_folder(uploads_folder)

#default app route (inital page load)
@app.route('/')
def index():
    return render_template('index.html')

#this is the transcribe page load
@app.route('/upload', methods=['POST'])
#have to function under app route
#function that runs when you click upload
def upload():
    #checking you've upload a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']

    #checking the file has a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    #making sure its an allowed extension
    if file and allowed_file(file.filename):
        #setting the file name as just the filename, not the path
        filename = secure_filename(file.filename)

        #clearing uploads folder when we have a new upload (in case anything left in it)
        clear_folder(uploads_folder)
        #Clear static/audio folder on upload of new file
        clear_folder('static/audio')

        #saving the file to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # #get the toggle state
        # toggle_state = request.form.get('toggle', 'false')
        # if toggle_state == 'false':
        #     run_nr = False
        # else:
        #     run_nr = True

        overall_type = request.form.get('selectedOptionNoise')

        if overall_type == "no":
            nr_type, option_num = None, "0"
        elif overall_type == 'dfn':
            nr_type, option_num = 'dfn', "0"
        elif overall_type == 'sg':
            nr_type, option_num = 'sg', "0"
        else:
            nr_type, option_num = 'dfn', "1"
      
        global full_text
        global source_language
        #running the end to end pipeline
        # data_for_csv, source_language, timestamps, full_text = e2e_pipeline(input_file_path=file_path, noise_reduction = run_nr, save=False)
        combined_timestamps, source_language, full_text = e2e_pipeline(input_file_path=file_path, noise_reduction = nr_type , save=True, option = option_num)
        #specifying the name of the converted audio file (in uploads folder)
        converted_audio_file_name = '{}_converted.wav'.format(str(filename).split('.')[0])
        #moving the converted audio file to the static/audio folder
        copy_wav_file(converted_audio_file_name)

        #clearing uploads folder
        clear_folder(uploads_folder)

        #returning the transcripted text
        # print(jsonify({'output': timestamps}))
        return jsonify({'output': combined_timestamps})
    else:
        return jsonify({'Intput': 'Error, please try again with a different file extension'})
        #ajax means communication between fe and be has to be in json files

#TODO: these are buttons that will appear after transcription is finished. This on will translate
@app.route('/translate', methods=['POST'])
def translate_web():
    selected_option = str(request.form.get('selectedOption'))
    print(selected_option, file=sys.stderr)
    # selected_option = 'dutch'

    global full_text
    global source_language

    translated = nllb_translate(full_text.replace("'", "").replace('"',''), source_language_input=source_language, target_language_input=selected_option)

    translated_language = selected_option

    return jsonify({'translated': str(translated), 'dest_lang': translated_language})

@app.route('/download_csv')
def download_csv():
    file_path = 'static/downloads/output.csv'
    return send_file(file_path, as_attachment=True)

#set a variable here depending on vm or locally
#if local - do port 5000 or 3000 say and if on VM, do port 50001

#hacky solution:

#running the website
if __name__ == '__main__':
    if torch.cuda.is_available():
        #VM
        app.run(debug=True, port=50001)
    else:
        #local
        app.run(debug=True, port=3000)