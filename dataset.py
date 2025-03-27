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
import torchaudio
from torchaudio.models.decoder import cuda_ctc_decoder
import Levenshtein

from sklearn.metrics import accuracy_score
import gc


from tqdm.auto import tqdm
import os


from config import config
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)


CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" :
     "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}


CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict[:-2] #To be used for mapping original transcripts to integer indices
LABELS = ARPAbet[:-2] #To be used for mapping predictions to strings

OUT_SIZE = len(PHONEMES) # Number of output classes

# Indexes of BLANK and SIL phonemes
BLANK_IDX=CMUdict.index('')
SIL_IDX=CMUdict.index('[SIL]')

class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    def __init__(self, root_dir, partition, subset):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        self.PHONEMES = PHONEMES
        self.subset = subset
        if partition == 'dev-clean':
            self.val_dataset = True
        else:
            self.val_dataset = False

        # Define the directories containing MFCC and transcript files
        self.mfcc_dir = os.path.join(config['root_dir'], partition, 'mfcc')
        self.transcript_dir = os.path.join(config['root_dir'], partition, 'transcript')

        # List all files in the directories. Remember to sort the files
        mfcc_names = os.listdir(self.mfcc_dir)
        transcript_names =   os.listdir(self.transcript_dir)
        mfcc_names.sort()
        transcript_names.sort()
        
        #if partition == "dev-clean":
        #    self.subset = 1.0
        
        # Compute size of data subset
        subset_size = int(self.subset * len(mfcc_names))

        # Select subset of data to use
        
        mfcc_names = mfcc_names[:subset_size]
        transcript_names = transcript_names[:subset_size]

        assert(len(mfcc_names) == len(transcript_names))

        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        self.length = len(mfcc_names) 

        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=config["augment_time_mask_max"]) #was 10 in original run
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=config["augment_freq_mask_max"]) #was 10 in original run
        #TODO
        # CREATE AN ARRAY TO STORE ALL PROCESSED MFCCS AND TRANSCRIPTS
        # LOAD ALL MFCCS AND CORRESPONDING TRANSCRIPTS AND DO THE NECESSARY PRE-PROCESSING
          # HINTS:
          # WHAT NORMALIZATION TECHNIQUE DID YOU USE IN HW1? CAN WE USE IT HERE?
          # REMEMBER TO REMOVE [SOS] AND [EOS] FROM TRANSCRIPTS
          
        self.mfccs = []
        self.transcripts = []
        for i in tqdm(range(len(mfcc_names))):

            # Load a single mfcc. Hint: Use numpy
            assert transcript_names[i] == mfcc_names[i] #check name of the two file is the same
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
          
            # Do Cepstral Normalization of mfcc along the Time Dimension (Think about the correct axis)
            # Hint: You may want to use np.mean and np.std
            mfcc_normalized = (mfcc - np.mean(mfcc)) / np.std(mfcc)
            # Convert mfcc to tensor
            mfcc_normalized = torch.tensor(mfcc_normalized, dtype=torch.float32)

            # Load the corresponding transcript
            # Remove [SOS] and [EOS] from the transcript
            # (Is there an efficient way to do this without traversing through the transcript?)
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.
            transcript  = np.load(os.path.join(self.transcript_dir, transcript_names[i]))
            
            transcript = transcript[1:-1]

            # The available phonemes in the transcript are of string data type
            # But the neural network cannot predict strings as such.
            # Hence, we map these phonemes to integers

            # Map the phonemes to their corresponding list indexes in self.phonemes
            #creating phonemes dictionary for efficient index lookup
            PHONEMES_DICT = {value: idx for idx, value in enumerate(PHONEMES)}
    
            transcript_indices = [PHONEMES_DICT[transcript_phonemes] for transcript_phonemes in transcript]
            # Now, if an element in the transcript is 0, it means that it is 'SIL' (as per the above example)

            # Convert transcript to tensor
            transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)
            # Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc_normalized)
            self.transcripts.append(transcript_indices)
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''
        self.length = len(self.mfccs)
       


    def __len__(self):

        return self.length


    def __getitem__(self, ind):

        '''
        RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.

        '''

        mfcc = self.mfccs[ind]  
        transcript = self.transcripts[ind]

        # NOTE: Remember to convert mfcc and transcripts to tensors here, if not done already in __init__
        return mfcc, transcript


    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''

        # Extract batch of input MFCCs and batch of output transcripts separately
        # Extract features and labels from batch
        batch_mfcc = [sample[0] for sample in batch]
        batch_transcript = [sample[1] for sample in batch]

        # Store original lengths of the MFCCS and transcripts in the batches
        # Store original lengths    
        lengths_mfcc = [mfcc.shape[0] for mfcc in batch_mfcc]  # Time dimension
        lengths_transcript = [transcript.shape[0] for transcript in batch_transcript]

        # Pad the MFCC sequences and transcripts
        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        # Note: (resulting shape of padded MFCCs: [batch, time, freq])
        
        #pad all sequences to longest sequence in the batch
        
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)
    


        # TODO: You may apply some transformations, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?
        #                     -> Time & Freq. Masking functions both expect input of shape (..., freq, time),
        #                        So permute your input dimensions appropriately before & after using these functions.

        #apply time masking and frequency masking
        if self.val_dataset == False:
            if random.random() < config["augment_prob"]:
                batch_mfcc_pad = batch_mfcc_pad.permute(0,2,1)
                batch_mfcc_pad = self.freq_masking(batch_mfcc_pad)
                batch_mfcc_pad = self.time_masking(batch_mfcc_pad)
                batch_mfcc_pad = batch_mfcc_pad.permute(0,2,1)
       

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc).cpu(), torch.tensor(lengths_transcript).cpu()


class AudioDatasetTest(torch.utils.data.Dataset):
    
    def __init__(self):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        self.PHONEMES = PHONEMES


        # Define the directories containing MFCC and transcript files
        self.mfcc_dir = os.path.join(config['root_dir'], "test-clean", 'mfcc')

        # List all files in the directories. Remember to sort the files
        mfcc_names = os.listdir(self.mfcc_dir)
        mfcc_names.sort()
        
        #if partition == "dev-clean":
        #    self.subset = 1.0
        
        # Compute size of data subset
       
        # Select subset of data to use
        
        mfcc_names = mfcc_names
       
        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        self.length = len(mfcc_names) 
   
        self.mfccs = []

        for i in tqdm(range(len(mfcc_names))):

            # Load a single mfcc. Hint: Use numpy
           
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
          
            # Do Cepstral Normalization of mfcc along the Time Dimension (Think about the correct axis)
            # Hint: You may want to use np.mean and np.std
            mfcc_normalized = (mfcc - np.mean(mfcc)) / np.std(mfcc)
            # Convert mfcc to tensor
            mfcc_normalized = torch.tensor(mfcc_normalized, dtype=torch.float32)

            # Load the corresponding transcript
            # Remove [SOS] and [EOS] from the transcript
            # (Is there an efficient way to do this without traversing through the transcript?)
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.
        
            # The available phonemes in the transcript are of string data type
            # But the neural network cannot predict strings as such.
            # Hence, we map these phonemes to integers

            # Map the phonemes to their corresponding list indexes in self.phonemes
            #creating phonemes dictionary for efficient index lookup
           
            # Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc_normalized)
    
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''
        self.length = len(self.mfccs)
        
    
    def __len__(self):

        return self.length


    def __getitem__(self, ind):

        '''
        RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.

        '''

        mfcc = self.mfccs[ind]  
        
        return (mfcc,)


    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''

        # Extract batch of input MFCCs and batch of output transcripts separately
        # Extract features and labels from batch
        batch_mfcc = [sample[0] for sample in batch]
       
        # Store original lengths of the MFCCS and transcripts in the batches
        # Store original lengths    
        lengths_mfcc = [mfcc.shape[0] for mfcc in batch_mfcc]  # Time dimension
       
        # Pad the MFCC sequences and transcripts
        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        # Note: (resulting shape of padded MFCCs: [batch, time, freq])
        
        #pad all sequences to longest sequence in the batch
        
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
    
        # TODO: You may apply some transformations, Time and Frequency masking, here in the collate function;
        # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
        #                  -> Would we apply transformation on the validation set as well?
        #                  -> Is the order of axes / dimensions as expected for the transform functions?
        #                     -> Time & Freq. Masking functions both expect input of shape (..., freq, time),
        #                        So permute your input dimensions appropriately before & after using these functions.

        return batch_mfcc_pad, torch.tensor(lengths_mfcc).cpu()



if __name__ == "__main__":
    
    train_data = AudioDataset(config['root_dir'], 'train-clean-100', 0.0001)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=train_data.collate_fn, num_workers=12)
    val_data = AudioDataset(config['root_dir'], 'dev-clean', subset=0.01)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, collate_fn=val_data.collate_fn, num_workers=12)
    
    print("Batch size: ", config['batch_size'])
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    
    x, y, lx, ly = next(iter(train_loader))
    print("Shape of x: ", x.shape)
    print("Shape of y: ", y.shape)
    print("lx: ", lx)
    print("ly: ", ly)