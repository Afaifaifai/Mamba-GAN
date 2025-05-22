wget https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0-midi.zip
unzip maestro-v1.0.0-midi.zip

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py37

python3 music_encoder.py --encode_official_maestro \  
    --mode midi_to_npy \  
    --pitch_transpose_lower -3 \  
    --pitch_transpose_upper 3 \  
    --output_folder ./maestro_magenta_s5_t3