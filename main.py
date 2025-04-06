from tabnanny import check
import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import wandb

from torchaudio.models.decoder import cuda_ctc_decoder

from tqdm.auto import tqdm
import os
import datetime

from config import config
import warnings
warnings.filterwarnings('ignore')

#set wandb environment variable
os.environ["WANDB_DIR"]= "/tmp"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

from ASR_model import ASRModel
from dataset import AudioDataset
from BasicNetwork import BasicNetwork

from phonemes_utils import PHONEMES, LABELS, BLANK_IDX
from ASR_model_large import ASRModelLarge
from train_utils import train_model, validate_model

train_data = AudioDataset(config['root_dir'], 'train-clean-100', config['subset'])
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=train_data.collate_fn, num_workers=128)
        
val_data = AudioDataset(config['root_dir'], 'dev-clean', 1)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, collate_fn=val_data.collate_fn, num_workers=128)       
        
print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))



#model = ASRModel(
#    input_size  = config["mfcc_features"],  #TODO,
#    embed_size  = config["embed_size"], #TODO
#    lstm_hidden_size=config["lstm_hidden_size"], 
#    output_size = len(PHONEMES)
#).to(device)

model = ASRModelLarge(
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

#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=config["learning_rate_min"], last_epoch=-1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config["epochs"], eta_min=config["learning_rate_min"], last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.8, patience=5, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=config["learning_rate_min"], eps=1e-8)
# Mixed Precision, if you need it
scaler = torch.cuda.amp.GradScaler()

RESUME_TRAINING = True
# If you are resuming an old run
if config["use_wandb"]:

    # Create your wandb run

    run_name = 'ASR_model_large-{}'.format(datetime.datetime.now().strftime("%m%d-%H%M%S"))
    
    wandb.login(key="11902c0c8e2c6840d72bf65f04894b432d85f019") #TODO

    if RESUME_TRAINING:
        run = wandb.init(
            id     = "xi0eiot4", ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs
            project = "hw3p2", ### Project should be created in your wandb
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



def load_model(path, model, optimizer= None, scheduler= None, scaler = None, metric='valid_dist'):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler != None:      
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    print("\nResuming training from epoch:", epoch)
    print('----------------------------------------\n')
    print("Epochs left: ", config['epochs'] - epoch)
    print("Optimizer: \n", optimizer)

    print("Best Val Dist:", metric)

    return [model, optimizer, scheduler, epoch, metric]


last_epoch_completed = 0
best_lev_dist = float('inf')


if RESUME_TRAINING:
    checkpoint_best_model_filename = 'checkpoint-model.pth_ASR_model_large-0328-151327'
    best_model_path = os.path.join(checkpoint_root, checkpoint_best_model_filename)
    model, optimizer, _, last_epoch_completed, best_lev_dist = load_model(best_model_path, model, optimizer, None, scaler, 'valid_dist')

#checkpoint_model_name = f'checkpoint-model.pth_{run_name}'
checkpoint_model_name = "checkpoint-model.pth_ASR_model_large-0328-151327"
best_model_path = os.path.join(checkpoint_root, checkpoint_model_name)

#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#    optimizer, T_0=5, T_mult=1, eta_min=1e-7, last_epoch=last_epoch_completed-1 
#)

for epoch in range(last_epoch_completed, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch + 1, config['epochs']))

    curr_lr = optimizer.param_groups[0]['lr']

    train_loss = train_model(model, train_loader, criterion, optimizer, scaler, device)
    valid_loss, valid_dist = validate_model(model, val_loader, decoder, device, criterion, LABELS)


    scheduler.step(valid_dist)

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
            #save wandb artifact
            artifact = wandb.Artifact(f'best_model_{run_name}', type='model')
            artifact.add_file(best_model_path)
            run.log_artifact(artifact)
            
            if config['use_wandb']:
                wandb.save(best_model_path)
            print("Saved best val model")
        
