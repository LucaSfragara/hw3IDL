import torch
import torch.nn as nn


class BasicNetwork(nn.Module):

    def __init__(self, input_dim=28, hidden_dim=256, out_size=41):

        super(BasicNetwork, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.2)

        self.classification = nn.Sequential(
            #TODO: Linear layer with in_features from the lstm module above and out_features = OUT_SIZE
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because of bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_size)
        )

        self.logSoftmax = nn.LogSoftmax(dim=-1)  # Apply log softmax along the class dimension

    def forward(self, x, lx):
        # x shape: [batch_size, seq_len, input_dim = mfcc_bins]
        # lx: sequence lengths
        
        # Transpose for CNN: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply CNN embedding
        x = self.embedding(x)  # [batch_size, hidden_dim, seq_len]
        
        # Transpose back for LSTM: [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Pack padded sequence for efficient LSTM processing
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lx.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Apply LSTM
        packed_out, _ = self.lstm(packed_x)
        
        # Unpack the sequence
        lstm_out, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Apply classification layer
        logits = self.classification(lstm_out)
        
        # Apply log softmax
        log_probs = self.logSoftmax(logits)
        
        return log_probs, output_lengths
