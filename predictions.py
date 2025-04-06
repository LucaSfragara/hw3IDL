import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchaudio.models.decoder import cuda_ctc_decoder
from config import config
from dataset import AudioDatasetTest, AudioDataset
from ASR_model_large import ASRModelLarge
import os
from phonemes_utils import PHONEMES, LABELS
from decode_utils import decode_prediction

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

model = ASRModelLarge(
    input_size  = config["mfcc_features"],  #TODO,
    embed_size  = config["embed_size"], #TODO
    lstm_hidden_size=config["lstm_hidden_size"], 
    output_size = len(PHONEMES)
).to(device)

test_decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['test_beam_width']) # TODO

results = []

audio_test_dataset = AudioDatasetTest()
test_loader = DataLoader(audio_test_dataset, batch_size=128, shuffle=False, collate_fn=audio_test_dataset.collate_fn, num_workers=12)



loaded_state = torch.load("checkpoints/checkpoint-model.pth_ASR_model_large-0328-151327")
print(loaded_state["valid_dist"])
model.load_state_dict(loaded_state['model_state_dict'])

model.eval()
print("Testing")

for data in tqdm(test_loader):

    x, lx   = data
    x, lx   = x.to(device), lx.to(device)

    with torch.no_grad():
        h, lh = model(x, lx)

    prediction_string = decode_prediction(h, lh.to(device), test_decoder, LABELS) # TODO call decode_prediction

    #TODO save the output in results array.
    # Hint: The predictions of each mini-batch are a list, so you may want to extend the results list, instead of append
    results.extend(prediction_string)
    del x, lx, h, lh
    torch.cuda.empty_cache()
    
if results:
    df = pd.DataFrame({
        'index': range(len(results)), 'label': results
    })

data_dir = "submission.csv"
df.to_csv(data_dir, index = False)

#run terminal command
os.system("kaggle competitions submit -c hw-3-p-2-automatic-speech-recognition-asr-11-785 -f submission.csv -m 'hello'")
