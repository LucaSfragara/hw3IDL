
import torch
import numpy as np

import torch.nn as nn

from torchinfo import summary

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from phonemes_utils import PHONEMES

import warnings
warnings.filterwarnings('ignore')

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print("Device: ", device)


class LockedDropout(torch.nn.Module):
    
    def __init__(self, p):
        super(LockedDropout, self).__init__()
        self.p = p
    def forward(self, x):
        
    
        if not self.training or self.p == 0:
            return x
        
        #check if x is a packed sequence
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            unpacked_x, x_lens = pad_packed_sequence(x, batch_first=True)
            
            # Apply locked dropout to unpacked tensor
            batch_size = unpacked_x.size(0)
            feature_size = unpacked_x.size(2)
            
            # Create mask: same for all timesteps in a sequence
            mask = torch.ones(batch_size, 1, feature_size, device=unpacked_x.device)
            mask = mask.bernoulli_(1 - self.p) / (1 - self.p)
            mask = mask.expand_as(unpacked_x)
            
            # Apply mask
            unpacked_x = unpacked_x * mask
            
            # Re-pack the sequence
            packed_x = pack_padded_sequence(
                unpacked_x, 
                x_lens.cpu(), 
                batch_first=True,
                enforce_sorted=False
            )
            return packed_x
        
        else: 
            batch_size = x.size(0)
            feature_size = x.size(2)
            
            mask =  torch.ones_like(x[:, 0, :], device=x.device).bernoulli_(1 - self.p) / (1 - self.p)
            mask = mask.unsqueeze(1).expand_as(x)
            return x * mask    


class Permute(torch.nn.Module):
    '''
    Used to transpose/permute the dimensions of an MFCC tensor.
    '''
    def forward(self, x):
        return x.transpose(1, 2)
    
class LSTMWrapper(torch.nn.Module):
    '''
    Used to get only output of lstm, not the hidden states.
    '''
    def __init__(self, lstm):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm

    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    
    
class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input?
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(
            input_size=input_size*2, #we are concatenating the feature dimension as we are reducing the time dimension by 2
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True  # Assuming your input is in batch_first format
    ) #  Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def forward(self, x_packed):
        
        # 1. Unpack the packed sequence
        x_padded, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        
        # 2. Downsample and reshape
        x_reshaped, x_lens_reshaped = self.trunc_reshape(x_padded, x_lens)
        
        # 3. Re-pack the sequence
        x_packed_reshaped = pack_padded_sequence(
            x_reshaped, 
            x_lens_reshaped, 
            batch_first=True,
            enforce_sorted=False  # Set to False if your sequences aren't sorted by length
        )
        
        # 4. Pass through the BLSTM layer
        output_packed, _ = self.blstm(x_packed_reshaped)
    
        # Return the packed output
        return output_packed

    def trunc_reshape(self, x, x_lens):

        batch_size, max_len, feat_dim = x.shape
    
        # Make sure the length is even (for downsampling by factor of 2)
        
        if max_len % 2 != 0:
            x = x[:, :-1, :]  # Remove last timestep if odd
            # Adjust lengths for sequences that used the last timestep
            x_lens = torch.where(x_lens > max_len - 1, x_lens - 1, x_lens)
        
        # Reshape: reduce timesteps by factor of 2, double features
        # [B, T, F] -> [B, T/2, 2F]
        new_max_len = max_len // 2
        x_reshaped = x.contiguous().view(batch_size, new_max_len, feat_dim * 2)
        
        # Reduce lengths by factor of 2 (integer division)
        x_lens_reshaped = torch.div(x_lens, 2, rounding_mode='floor')
        
        return x_reshaped, x_lens_reshaped
    
    
class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.gelu = torch.nn.GELU()
        self.dropout1 = torch.nn.Dropout(0.1, inplace=True)
        
        # Second convolutional layer
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.dropout2 = torch.nn.Dropout(0.1, inplace=True)
        
        # Optional 1x1 convolution for matching dimensions if needed
        self.residual = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        # Save the input for the skip connection
  

        residual = x if self.residual is None else self.residual(x)
        
        # Pass through the first convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        
        # Pass through the second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Add the skip connection
        x += residual
        x = self.gelu(x)
  
        return x
        

class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, embedding_hidden_size, lstm_hidden_size):
        super(Encoder, self).__init__()


        #You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.
        # Food for thought -> What type of Conv layers can be used here?
        #                  -> What should be the size of input channels to the first layer?
        self.embedding = torch.nn.Sequential(
            
            torch.nn.Conv1d(input_size, 64, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1, inplace=True),
            
            ResNetBlock(in_channels=64, out_channels=64),
            ResNetBlock(in_channels=64, out_channels=128),
            ResNetBlock(in_channels=128, out_channels=embedding_hidden_size),
            ResNetBlock(in_channels=embedding_hidden_size, out_channels=embedding_hidden_size),
            ResNetBlock(in_channels=embedding_hidden_size, out_channels=embedding_hidden_size),
            ResNetBlock(in_channels=embedding_hidden_size, out_channels=embedding_hidden_size),
        )

        self.BLSTMs = LSTMWrapper(
            # TODO: Look up the documentation. You might need to pass some additional parameters.
            torch.nn.LSTM(input_size=embedding_hidden_size, hidden_size=lstm_hidden_size, num_layers=4, bidirectional=True) #TODO
        )

        self.pBLSTMs = torch.nn.Sequential( # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be?
            # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
            # https://github.com/salesforce/awd-lstm-lm/blob/dfd3cb0235d2caf2847a4d53e1cbd495b781b5d2/locked_dropout.py#L5
            # ...
            pBLSTM(2* lstm_hidden_size, lstm_hidden_size),
            LockedDropout(0.2),
            pBLSTM(2* lstm_hidden_size, lstm_hidden_size),
            LockedDropout(0.2),
            #pBLSTM(2* lstm_hidden_size, lstm_hidden_size),
            #LockedDropout(0.2),
        )

    def forward(self, x, x_lens):
        
        # Where are x and x_lens coming from? The dataloader
        # shape of x: (B, T, MFCC_bins=28)

        # Transpose to (B, MFCC_bins, T) for Conv1d.
        x = x.transpose(1, 2)
        x = self.embedding(x)  # (B, 64, T)
        
        # Transpose back to (B, T, 64) for LSTM.
        x = x.transpose(1, 2)
        x_lens = x_lens.cpu()
        # Pack the padded sequence.
        x_packed = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True,
                                                     enforce_sorted=False)
        # TODO: Pass Sequence through the Bi-LSTM layer
        x_blstm = self.BLSTMs(x_packed) # Also outputs packed sequence
        # TODO: Pass Sequence through the pyramidal Bi-LSTM layer
        encoder_outputs = self.pBLSTMs(x_blstm)
        # TODO: Pad Packed Sequence
        encoder_outputs, encoder_lens = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)
        # Remember the number of output(s) each function returns ???

        return encoder_outputs, encoder_lens


class Decoder(torch.nn.Module):

    def __init__(self, lstm_hidden_size, output_size= 41):
        super().__init__()

        self.mlp = torch.nn.Sequential(

            Permute(),
            torch.nn.BatchNorm1d(2 * lstm_hidden_size),
            Permute(),

            #TODO define your MLP arch. Refer HW1P2
            #Use Permute Block before and after BatchNorm1d() to match the size
            #Now you can stack your MLP layers
            # MLP layers with dropout for regularization
            
            torch.nn.Linear(2 * lstm_hidden_size, 2*lstm_hidden_size),
            Permute(),
            torch.nn.BatchNorm1d(2*lstm_hidden_size),
            Permute(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(2*lstm_hidden_size, 2*lstm_hidden_size),
            Permute(),
            torch.nn.BatchNorm1d(2*lstm_hidden_size),
            Permute(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(2*lstm_hidden_size, lstm_hidden_size),
            Permute(),
            torch.nn.BatchNorm1d(lstm_hidden_size),
            Permute(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            
            
            torch.nn.Linear(lstm_hidden_size, lstm_hidden_size),
            Permute(),
            torch.nn.BatchNorm1d(lstm_hidden_size),
            Permute(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(lstm_hidden_size, lstm_hidden_size),
            Permute(),
            torch.nn.BatchNorm1d(lstm_hidden_size),
            Permute(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            
            # Final projection layer to output_size (number of phonemes)
            torch.nn.Linear(lstm_hidden_size, output_size)

        )

        self.softmax = torch.nn.LogSoftmax(dim=2)


    def forward(self, encoder_out):

        #TODO: Call your MLP
        
        out = self.mlp(encoder_out)  
        log_probs = self.softmax(out)

        #TODO: Think about what should be the final output of the decoder for classification

        return log_probs
    
class ASRModelLarge(torch.nn.Module):

    def __init__(self, input_size, embed_size= 192, lstm_hidden_size = 64, output_size= len(PHONEMES)):
        super().__init__()

        # Initialize encoder and decoder

        self.encoder        = Encoder(input_size=input_size, embedding_hidden_size= embed_size, lstm_hidden_size=lstm_hidden_size) # TODO: Initialize Encoder
        self.decoder        = Decoder(lstm_hidden_size=lstm_hidden_size, output_size=output_size) # TODO: Initialize Decoder


    def forward(self, x, lengths_x):

        encoder_out, encoder_lens   = self.encoder(x, lengths_x)
        decoder_out                 = self.decoder(encoder_out)

        return decoder_out, encoder_lens
    


if __name__ == "__main__":
    
    #get model number of params
    model = ASRModelLarge(input_size=28, embed_size=256, lstm_hidden_size=256)
    x = torch.tensor(np.random.rand(10, 100, 28)).float()
    lx = torch.tensor([100]*10)
    print(summary(model, input_data = [x, lx]))