config = {
  "subset": 1,
  "learning_rate": 1e-3,
  "learning_rate_min": 1e-6,
  "epochs": 160,
  "train_beam_width": 5,
  "test_beam_width": 2,
  "mfcc_features": 28,
  "embed_size": 350,
  "lstm_hidden_size": 350,
  "batch_size": 128,
  "encoder_dropout": 0.2,
  "lstm_dropout": 0.2,
  "decoder_dropout": 0.2,
  "use_wandb": True,
  "root_dir": "/local/11785-S25-hw3p2",
  "name": "Luca Sfragara",  
  "weight_decay": 0.0001,
  "augment_prob": 0.7,
  "augment_freq_mask_max":20, 
  "augment_time_mask_max":20,
}
