import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import wandb
from torchaudio.models.decoder import cuda_ctc_decoder
from decode_utils import decode_prediction, calculate_levenshtein
from train_utils import train_model, validate_model

from tqdm import tqdm
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

from phonemes_utils import PHONEMES, LABELS

train_data = AudioDataset(config['root_dir'], 'train-clean-100', config['subset'])
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=train_data.collate_fn, num_workers=12)
        
val_data = AudioDataset(config['root_dir'], 'dev-clean', 1)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, collate_fn=val_data.collate_fn, num_workers=12)       
        
print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))



wandb.login(key="11902c0c8e2c6840d72bf65f04894b432d85f019") #TODO



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

checkpoint_best_model_filename = 'checkpoint-best-model.pth'
best_model_path = os.path.join(checkpoint_root, checkpoint_best_model_filename)


sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'valid_dist',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3, 1e-4, 1e-5]
        },
        'min_learning_rate_factor': {
            'values': [10, 100]
        },
        'lstm_hidden_size': {
            'values': [128, 256]
        },
        'embed_size': {
            'values': [64, 128],
        }, 
    }
}

sweep_id = wandb.sweep(sweep_config, project="hw3p2")


def run_sweep():
    
    run = wandb.init()
    run_config = run.config
    
    #unpack config
    learning_rate = run_config["learning_rate"]
    min_learning_rate_factor = run_config["min_learning_rate_factor"]
    min_learning_rate = learning_rate / min_learning_rate_factor
    lstm_hidden_size = run_config["lstm_hidden_size"]
    embed_size = run_config["embed_size"]
    
    x, y, lx, ly = next(iter(train_loader))

    
    model = ASRModel(
        input_size  = config["mfcc_features"],  #TODO,
        embed_size  = embed_size, #TODO
        lstm_hidden_size=lstm_hidden_size, 
        output_size = len(PHONEMES)
    ).to(device)
    
    print(summary(model, input_data=[x.to(device), lx.to(device)]))
    
    criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True) 

    optimizer =  torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=config["weight_decay"]) #What goes in here?

    decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['train_beam_width']) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=min_learning_rate, last_epoch=-1)
    scaler = torch.cuda.amp.GradScaler()

    wandb.watch(model, log="all")
    
    ### Save your model architecture as a string with str(model)
    model_arch  = str(model)
    ### Save it in a txt file
    arch_file   = open("model_arch.txt", "w")
    file_write  = arch_file.write(model_arch)
    arch_file.close()

    ### log it in your wandb run with wandb.save()
    wandb.save('model_arch.txt')
    last_epoch_completed = 0
    best_lev_dist = float("inf")
    for epoch in range(last_epoch_completed, config['epochs']):

        print("\nEpoch: {}/{}".format(epoch + 1, config['epochs']))

        curr_lr = optimizer.param_groups[0]['lr']


        train_loss = train_model(model, train_loader, criterion, optimizer, scaler, device)
        valid_loss, valid_dist = validate_model(model, val_loader, decoder, device, criterion,LABELS)

        scheduler.step()

        print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
        print("\tVal Dist {:.04f}\t Val Loss {:.04f}".format(valid_dist, valid_loss))



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
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file(best_model_path)
            run.log_artifact(artifact)
            
        
            


wandb.agent(sweep_id, function=run_sweep)
    
