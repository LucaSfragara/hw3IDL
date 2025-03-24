from symbol import comp_for
from turtle import mode
import torch
import random
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import wandb
import torchaudio.transforms as tat
from torchaudio.models.decoder import cuda_ctc_decoder
import Levenshtein

from sklearn.metrics import accuracy_score
import gc

import glob

import zipfile
from tqdm.auto import tqdm
import os
import datetime

from config import config
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

from ASR_model import ASRModel
from dataset import AudioDataset
from BasicNetwork import BasicNetwork

from phonemes_utils import PHONEMES, LABELS, BLANK_IDX

train_data = AudioDataset(config['root_dir'], 'train-clean-100', config['subset'])
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=train_data.collate_fn, num_workers=128)
        
val_data = AudioDataset(config['root_dir'], 'dev-clean', 1)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, collate_fn=val_data.collate_fn, num_workers=128)       
        
print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))



model = ASRModel(
    input_size  = config["mfcc_features"],  #TODO,
    embed_size  = config["embed_size"], #TODO
    lstm_hidden_size=config["lstm_hidden_size"], 
    output_size = len(PHONEMES)
).to(device)

#model = BasicNetwork(input_dim=config["mfcc_features"], hidden_dim=config["embed_size"], out_size=len(LABELS)).to(device)

x, y, lx, ly = next(iter(train_loader))

print(summary(model, input_data=[x.to(device), lx.to(device)]))


# Define CTC loss as the criterion. How would the losses be reduced?
criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True) # What goes in here?
# CTC Loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
# Refer to the handout for hints

optimizer =  torch.optim.AdamW(model.parameters(), lr = config["learning_rate"], weight_decay=config["weight_decay"]) #What goes in here?

# Declare the decoder. Use the PyTorch Cuda CTC Decoder to decode phonemes
# CTC Decoder: https://pytorch.org/audio/2.1/generated/torchaudio.models.decoder.cuda_ctc_decoder.html
decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['train_beam_width']) 

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=config["learning_rate_min"], last_epoch=-1)

# Mixed Precision, if you need it
scaler = torch.cuda.amp.GradScaler()


def decode_prediction(output, output_lens, decoder, PHONEME_MAP=LABELS):
    """
    Decode model output to phoneme strings using CTC decoder
    
    Args:
        output: Log probabilities from the model [B, T, V]
        output_lens: Lengths of each sequence in the batch
        decoder: CTC decoder instance
        PHONEME_MAP: Mapping from indices to phoneme characters
    """
    # Ensure output is contiguous
    output = output.contiguous()
    output_lens = output_lens.to(torch.int32).contiguous()
    beam_results = decoder(output, output_lens.to(torch.int32))
    pred_strings = []
    
    for i in range(len(beam_results)):
        
        top_beam_results = beam_results[i][0].tokens
        
        labels = [PHONEME_MAP[t] for t in top_beam_results]
        pred_string = ''.join(labels)
        pred_strings.append(pred_string)
    

    return pred_strings

def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP=LABELS):
    dist = 0
    batch_size = label.shape[0]

    pred_strings = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

    for i in range(batch_size):
        # Get the actual label string for this sample
        # Convert the label indices to phonemes and join them
        label_string = [PHONEME_MAP[t] for t in label[i][:label_lens[i]]]
        
        # Get the predicted string from decode_prediction
        pred_string = pred_strings[i]

        # Calculate Levenshtein distance between predicted and actual strings
        dist += Levenshtein.distance(pred_string, label_string)

    # Average the distance over the batch
    dist /= batch_size  # We average to get a normalized metric independent of batch size
    return dist


    
last_epoch_completed = 0
best_lev_dist = float("inf")


# Train function
def train_model(model, train_loader, criterion, optimizer):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        lx, ly = lx.to(device), ly.to(device)

        with torch.cuda.amp.autocast():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


# Eval function
def validate_model(model, val_loader, decoder, phoneme_map = LABELS):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        lx, ly = lx.to(device), ly.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh.to(device), ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist




# If you are resuming an old run
if config["use_wandb"]:

    RESUME_LOGGING = False # Set this to true if you are resuming training from a previous run

    # Create your wandb run

    run_name = 'ASR_model-{}'.format(datetime.datetime.now().strftime("%m%d-%H%M%S"))
    
    wandb.login(key="11902c0c8e2c6840d72bf65f04894b432d85f019") #TODO

    if RESUME_LOGGING:
        run = wandb.init(
            id     = "", ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs
            project = "hw3p2-ablations", ### Project should be created in your wandb
            settings = wandb.Settings(_service_wait=300)
        )
    else: 
        run = wandb.init(
            name    = run_name, ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True, ### Allows reinitalizing runs when you re-run this cell
            project = "hw3p2", ### Project should be created in your wandb account
            config  = config ### Wandb Config for your run
        )


    ### Save your model architecture as a string with str(model)
    model_arch  = str(model)
    ### Save it in a txt file
    arch_file   = open("model_arch.txt", "w")
    file_write  = arch_file.write(model_arch)
    arch_file.close()

    ### log it in your wandb run with wandb.save()
    wandb.save('model_arch.txt')


def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict() if scheduler is not None else {},
         'scaler_state_dict'        : scaler.state_dict() if scaler is not None else {},
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )

checkpoint_root = os.path.join(os.getcwd(), 'checkpoints')
os.makedirs(checkpoint_root, exist_ok=True)

if config["use_wandb"]:
    wandb.watch(model, log="all")

checkpoint_best_model_filename = 'checkpoint-best-model.pth'
best_model_path = os.path.join(checkpoint_root, checkpoint_best_model_filename)


for epoch in range(last_epoch_completed, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch + 1, config['epochs']))

    curr_lr = optimizer.param_groups[0]['lr']


    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_dist = validate_model(model, val_loader, decoder, LABELS)


    scheduler.step()

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}\t Val Loss {:.04f}".format(valid_dist, valid_loss))


    if config['use_wandb']:
        wandb.log({
                'train_loss': train_loss,
                'valid_dist': valid_dist,
                'valid_loss': valid_loss,
                'lr': curr_lr
        })
       
    
        if valid_dist <= best_lev_dist:
            best_lev_dist = valid_dist
            save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
            if config['use_wandb']:
                wandb.save(best_model_path)
            print("Saved best val model")
        
