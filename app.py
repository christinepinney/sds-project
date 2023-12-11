import os
import pandas as pd 
import numpy as np 
import flask
import pickle
from flask import Flask, render_template, request
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel
import torch
from os import path
from pydub import AudioSegment
from pydub.utils import which
import torchaudio

UPLOAD_FOLDER = 'uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = pickle.load(open('finalized_model.pkl', 'rb'))

def get_bert_embedding(text):

    ''' Convert text data to corresponding BERT embedding '''

    bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(ids).unsqueeze(0)
    tensor = ''
    with torch.no_grad():
        o = bert(input_ids)
        embedding = o.last_hidden_state[0]
        tensor = embedding[:1,:768]
        tensor = torch.Tensor(tensor)
    return tensor


def get_w2v_embedding(filename):

    ''' Convert audio data to corresponding Wav2Vec embedding '''

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()

    new_file = filename[:-4] + '.wav'
    filename = "uploads/" + filename
    AudioSegment.from_file(filename).export(new_file, format='wav')

    waveform, sample_rate = torchaudio.load(new_file)
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    tensor = ''
    with torch.inference_mode():
        emission, _ = model(waveform)
        tensor = emission[:1,0,:768]

    return tensor


@app.route('/')
def index():
    return flask.render_template('app.html')


@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        # convert text data to bert embedding
        text_data = request.form['textbox']
        bert_embed = get_bert_embedding(text_data)

        # convert audio data to wav2vec embedding
        audio_data = request.files['file']
        audio_data.save(os.path.join(app.config['UPLOAD_FOLDER'], audio_data.filename))
        w2v_embed = get_w2v_embedding(audio_data.filename)

        # concatenate embeddings into single tensor
        full = torch.cat((w2v_embed, bert_embed), dim=1)

        # send tensor through encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=1536, nhead=4)
        encode = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        encoded = encode(full)

        # and through decoder
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=1536, nhead=4)
        decode = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)
        decoded = decode(encoded, full)

        # put output in form that sklearn model can handle
        decode_out = decoded.detach().numpy()

        # predict sarcasm
        sarc = model.predict(decode_out)
        if int(sarc) == 1:
            prediction = 'Sarcasm detected!'
        else:
            prediction = 'No sarcasm detected.'
        return render_template('result.html', prediction = prediction)
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
