# '''''' Required Arguments ''''''
output_dir = 'output_dir'  # The output directory where the model predictions and checkpoints will be written.
train_dir = '../data/maestro_magenta_s5_t3/train'  # The output directory where the model predictions and checkpoints will be written.
eval_dir = '../data/maestro_magenta_s5_t3/valid'   # The output directory where the model predictions and checkpoints will be written.
test_dir = '../data/maestro_magenta_s5_t3/test'   # The output directory where the model predictions and checkpoints will be written.
vocab_file = '../data/maestro_magenta_s5_t3/vocab.txt'  # The vocab file.
event_type = 'magenta'  # The event type. Choices: ['magenta', 'newevent']
model_type = 'bert'  # The model architecture to be trained or fine-tuned.

# '''''' Model Architecture ''''''
num_hidden_layers = 5    # Number of layers in BERT.
hidden_size = 768        # The number of hidden space

# '''''' Checkpoint & Continuation ''''''
should_continue = False     # Whether to continue from latest checkpoint in output_dir
model_name_or_path = None   # The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.
config_name = None          # Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.
tokenizer_name = 'midi_tokenizer'  # Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.
cache_dir = None            # Optional directory to store the pre-trained models downloaded (instead of the default one)
block_size = 10             # Optional input sequence length after tokenization. Default to the model max input length for single sentence inputs.

# '''''' Training Mode ''''''
mlm = True                # Train with masked-language modeling loss instead of language modeling.
mlm_probability = 0.15     # Ratio of tokens to mask for masked language modeling loss

# '''''' Execution Flags ''''''
do_train = True                  # Whether to run training.
do_eval = True                   # Whether to run eval on the dev set.
evaluate_during_training = True  # Run evaluation during training at each logging step.

# '''''' Batch & Gradient ''''''
per_gpu_train_batch_size = 2048  # Batch size per GPU/CPU for training.
per_gpu_eval_batch_size = 2048   # Batch size per GPU/CPU for evaluation.
gradient_accumulation_steps = 1  # Number of update steps to accumulate before performing a backward/update pass.

# '''''' Optimization ''''''
learning_rate = 1e-4  # The initial learning rate for Adam.
weight_decay = 0.0    # Weight decay if we apply some.
adam_epsilon = 1e-8   # Epsilon for Adam optimizer.
max_grad_norm = 1.0   # Max gradient norm.
warmup_steps = 1000   # Linear warmup over warmup_steps.

# '''''' Training Schedule ''''''
num_train_epochs = 100000.0  # Total number of training epochs to perform.
max_steps = -1               # If > 0: set total number of training steps to perform. Override num_train_epochs.

# '''''' Logging & Saving ''''''
logging_steps = 5000         # Log every X update steps.
save_steps = 1000            # Save checkpoint every X update steps.
save_total_limit = 20        # Limit the total amount of checkpoints, delete the older ones in output_dir.
eval_all_checkpoints = False # Evaluate all checkpoints starting with the same prefix as model_name_or_path ending with step number.
overwrite_output_dir = True  # Overwrite the content of the output directory.
overwrite_cache = False      # Overwrite the cached training and evaluation sets.

# '''''' Device & Misc ''''''
no_cuda = False      # Avoid using CUDA when available.
local_rank = -1      # For distributed training: local_rank.
seed = 42            # Random seed for initialization.
fp16 = False         # Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.
fp16_opt_level = "O1"  # For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
