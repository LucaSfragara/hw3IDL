import torch
import Levenshtein
from phonemes_utils import PHONEMES, LABELS


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

