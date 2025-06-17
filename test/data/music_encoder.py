# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from performance_event_repo import PerformanceEventRepo
import functools
import time
import os, sys
import pandas as pd
import logging
sys.path.append(os.path.dirname(sys.path[0]))
from utils import find_files_by_extensions
from tqdm import tqdm
import glob
from datasets import Dataset, DatasetDict

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
MAESTOR_V1_DIR = os.path.join(_CURR_DIR, 'maestro-v1.0.0')

def read_maestro_meta_info(data_dir):
    """Read the meta information from Maestro

    Parameters
    ----------
    data_dir
        The base path of the maestro data

    Returns
    -------
    df
        Pandas Dataframe, with the following columns:
        ['canonical_composer',
         'canonical_title',
         'split',
         'year',
         'midi_filename',
         'audio_filename',
         'duration']
    """
    if os.path.exists(os.path.join(data_dir, 'maestro-v1.0.0.csv')):
        logging.info('Process maestro-v1.')
        csv_path = os.path.join(data_dir, 'maestro-v1.0.0.csv')
    elif os.path.exists(os.path.join(data_dir, 'maestro-v2.0.0.csv')):
        logging.info('Process maestro-v2.')
        csv_path = os.path.join(data_dir, 'maestro-v2.0.0.csv')
    else:
        raise ValueError('Cannot found valid csv files!')
    df = pd.read_csv(csv_path)
    return df


def get_midi_paths():
    if not os.path.exists(MAESTOR_V1_DIR):
        raise ValueError('Cannot find maestro-v1.0.0, use `get_data.sh` to download and '
                         'extract the data.')
    df = read_maestro_meta_info(MAESTOR_V1_DIR)
    train_paths = df[df['split'] == 'train']['midi_filename'].to_numpy()
    validation_paths = df[df['split'] == 'validation']['midi_filename'].to_numpy()
    test_paths = df[df['split'] == 'test']['midi_filename'].to_numpy()
    train_paths = [os.path.join(MAESTOR_V1_DIR, ele) for ele in train_paths]
    validation_paths = [os.path.join(MAESTOR_V1_DIR, ele) for ele in validation_paths]
    test_paths = [os.path.join(MAESTOR_V1_DIR, ele) for ele in test_paths]
    return train_paths, validation_paths, test_paths


if __name__ == '__main__':

    from argparse import ArgumentParser
    import multiprocessing as mpl

    parser = ArgumentParser()
    parser.add_argument('--input_folder', type=str,
                        help='Directory with the downloaded MAESTOR dataset',
                        default=MAESTOR_V1_DIR)
    parser.add_argument('--output_folder', type=str,
                        help='Directory to encode the event signals', required=True)
    parser.add_argument('--encode_official_maestro', action='store_true',
                        help='Whether to encode the official Maestro dataset.')
    parser.add_argument('--mode', type=str,
                        help='Convert to/from MIDIs to TXT/Numpy',
                        choices=['to_txt', 'to_midi', 'txt_to_alltxt', 'npy_to_allnpy',
                                 'midi_to_npy', 'npy_to_midi'],
                        default='to_txt')
    parser.add_argument('--stretch_factors', type=str, help='Stretch Factors',
                        default='0.95,0.975,1.0,1.025,1.05')
    parser.add_argument('--pitch_transpose_lower', type=int,
                        help='Lower bound of the pitch transposition amounts',
                        default=-3)
    parser.add_argument('--pitch_transpose_upper', type=int,
                        help='Uppwer bound of the pitch transposition amounts',
                        default=3)
    args = parser.parse_args()


    stretch_factors = [float(ele) for ele in args.stretch_factors.split(',')]
    encoder = PerformanceEventRepo(steps_per_second=100, num_velocity_bins=32,
                               stretch_factors=stretch_factors,
                               pitch_transpose_lower=args.pitch_transpose_lower,
                               pitch_transpose_upper=args.pitch_transpose_upper)

    def run_to_text(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.to_text(path, os.path.join(out_dir, filename + '.txt'))


    def run_to_text_with_transposition(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.to_text_transposition(path, os.path.join(out_dir, filename + '.txt'))


    def run_to_npy(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.to_npy(path, os.path.join(out_dir, filename + '.npy'))


    def run_to_npy_with_transposition(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.to_npy_transposition(path, os.path.join(out_dir, filename + '.npy'))


    def run_from_text(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.from_text(path, os.path.join(out_dir, filename + '.mid'))


    def run_npy_to_midi(path, out_dir):
        filename, extension = os.path.splitext(os.path.basename(path))
        encoder.npy_to_midi(path, os.path.join(out_dir, filename + '.mid'))

    def merge_txt_files_in_folder(source_folder, output_file):
        """
        掃描一個來源資料夾，找到其中所有的 .txt 檔案，
        將每個檔案的內容合併成一行，然後全部寫入一個大的輸出檔案。

        Args:
            source_folder (str): 包含多個 .txt 檔案的來源資料夾路徑。
            output_file (str): 合併後要儲存的單一檔案路徑。
        """
        # 使用 glob 找到所有 .txt 檔案的路徑
        txt_files = glob.glob(os.path.join(source_folder, '*.txt'))
        
        if not txt_files:
            print(f"警告：在 '{source_folder}' 中沒有找到任何 .txt 檔案。")
            return

        print(f"找到 {len(txt_files)} 個 .txt 檔案。開始從 '{source_folder}' 合併至 '{output_file}'...")
        
        # 打開輸出的檔案準備寫入
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # 使用 tqdm 顯示進度條
            for file_path in tqdm(txt_files, desc=f"合併 {os.path.basename(source_folder)}"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        # 讀取檔案中的所有行 (每個 token)，並移除換行符
                        tokens = [line.strip() for line in f_in.readlines()]
                        # 將所有 token 用空格連接成一個長字串
                        sequence_line = " ".join(tokens)
                        # 將這個代表單一序列的長字串寫入輸出檔案，並加上換行符
                        f_out.write(sequence_line + "\n")
                except Exception as e:
                    print(f"處理檔案 {file_path} 時發生錯誤: {e}")
                    
        print(f"成功！'{source_folder}' 中的所有 .txt 檔案已合併至 '{output_file}'。")

    def load_sequences_from_npy_folder(folder_path):
        """從一個資料夾中載入所有 .npy 檔案，並回傳一個包含所有序列的列表。"""
        npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
        all_sequences = []
        print(f"在 '{folder_path}' 中找到 {len(npy_files)} 個 .npy 檔案...")
        for file_path in tqdm(npy_files, desc=f"載入 {os.path.basename(folder_path)}"):
            try:
                # 載入 .npy 檔案，並將其轉換為 list
                sequence = np.load(file_path).tolist()
                all_sequences.append(sequence)
            except Exception as e:
                print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
        return all_sequences

    num_cpus = mpl.cpu_count()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.mode == 'to_txt' or args.mode == 'midi_to_npy':
        if args.mode == 'to_txt':
            converted_format = 'txt'
            convert_transposition_f = run_to_text_with_transposition
            convert_f = run_to_text
        else:
            converted_format = 'npy'
            convert_transposition_f = run_to_npy_with_transposition
            convert_f = run_to_npy

        print('Converting midi files from {} to {}...'
              .format(args.input_folder, converted_format))
        if args.encode_official_maestro:
            train_paths, valid_paths, test_paths = get_midi_paths()
            print('Load MAESTRO V1 from {}. Train/Val/Test={}/{}/{}'
                  .format(args.input_folder, len(train_paths), len(valid_paths),
                          len(test_paths)))
            for split_name, midi_paths in [('train', train_paths),
                                           ('valid', valid_paths),
                                           ('test', test_paths)]:
                if split_name == 'train':
                    convert_function = convert_transposition_f
                else:
                    convert_function = convert_f
                out_split_dir = os.path.join(args.output_folder, split_name)
                os.makedirs(out_split_dir, exist_ok=True)
                start = time.time()
                with mpl.Pool(num_cpus - 1) as pool:
                    pool.map(functools.partial(convert_function, out_dir=out_split_dir),
                             midi_paths)
                print('Split {} converted! Spent {}s to convert {} samples.'
                      .format(split_name, time.time() - start, len(midi_paths)))
            encoder.create_vocab_txt(args.output_folder)
        else:
            midi_paths = []
            for root, _, files in os.walk(args.input_folder):
                for fname in files:
                    filename, extension = os.path.splitext(os.path.basename(fname))
                    if extension == '.mid' or extension == '.midi':
                        midi_paths.append(os.path.join(root, fname))
            os.makedirs(args.output_folder, exist_ok=True)
            start = time.time()
            with mpl.Pool(num_cpus - 1) as pool:
                pool.map(functools.partial(convert_f, out_dir=args.output_folder),
                         midi_paths)
            print('Converted midi files from {} to {}! Spent {}s to convert {} samples.'
                  .format(args.input_folder, args.output_folder,
                          time.time() - start, len(midi_paths)))
    elif args.mode == 'to_midi' or args.mode == 'npy_to_midi':
        convert_f = run_from_text if args.mode == 'to_midi' else run_npy_to_midi
        start = time.time()
        if args.mode == 'npy_to_midi':
            input_paths = list(find_files_by_extensions(args.input_folder, ['.npy']))
        else:
            input_paths = list(find_files_by_extensions(args.input_folder, ['.txt']))
        with mpl.Pool(num_cpus - 1) as pool:
            pool.map(functools.partial(convert_f,
                                       out_dir=args.output_folder),
                     input_paths)
        print('Test converted! Spent {}s to convert {} samples.'
              .format(time.time() - start, len(input_paths)))
    elif args.mode == 'txt_to_alltxt':
        BASE_DATA_DIR = "./maestro_magenta_s5_t3" 

        # --- 定義來源資料夾和目標檔案 ---
        # key 是來源資料夾的名稱，value 是要輸出的目標檔案名稱
        split_config = {
            "train": "train_all_data.txt",
            "valid": "valid_all_data.txt",
            "test": "test_all_data.txt"
        }

        # --- 開始執行合併 ---
        for split_name, output_filename in split_config.items():
            # 組合出完整的來源資料夾路徑
            source_dir = os.path.join(BASE_DATA_DIR, split_name)
            # 組合出完整的輸出檔案路徑
            output_path = os.path.join(BASE_DATA_DIR, output_filename)
            
            # 呼叫合併函式
            merge_txt_files_in_folder(source_dir, output_path)
            print("-" * 50)

        print("所有數據集合併完成！")

    elif args.mode == 'npy_to_allnpy':
        # --- 請設定您的來源和目標路徑 ---
        # 包含 train, valid, test 三個子資料夾的 .npy 數據根目錄
        # 這應該是您原始 music_encoder.py 產生 .npy 的地方
        BASE_NPY_DIR = "./maestro_magenta_s5_t3" 
        
        # 您想要將最終處理好的 Arrow 格式數據集儲存到哪裡
        FINAL_DATASET_SAVE_PATH = "./arrow_dataset"
        # ------------------------------------

        print("開始從 .npy 檔案建立 Hugging Face Dataset...")

        # 分別為 train, valid, test 建立 Dataset 物件
        train_sequences = load_sequences_from_npy_folder(os.path.join(BASE_NPY_DIR, 'train'))
        valid_sequences = load_sequences_from_npy_folder(os.path.join(BASE_NPY_DIR, 'valid'))
        test_sequences = load_sequences_from_npy_folder(os.path.join(BASE_NPY_DIR, 'test'))

        # 將載入的序列轉換成 Dataset 要求的字典格式
        train_dataset = Dataset.from_dict({"input_ids": train_sequences})
        valid_dataset = Dataset.from_dict({"input_ids": valid_sequences})
        test_dataset = Dataset.from_dict({"input_ids": test_sequences})

        # 將三個 Dataset 物件打包成一個 DatasetDict
        raw_datasets = DatasetDict({
            'train': train_dataset,
            'validation': valid_dataset,
            'test': test_dataset
        })

        print("\n數據集結構預覽:")
        print(raw_datasets)
        
        # 將這個數據集字典以高效的 Arrow 格式儲存到硬碟
        print(f"\n正在將數據集儲存至 '{FINAL_DATASET_SAVE_PATH}'...")
        raw_datasets.save_to_disk(FINAL_DATASET_SAVE_PATH)

        print("-" * 50)
        print("🎉 成功！您的 .npy 數據已轉換為高效的 Arrow 數據集。")
        print(f"現在您可以在訓練指令中使用 '{FINAL_DATASET_SAVE_PATH}' 這個路徑了。")
        print("-" * 50)

    else:
        raise NotImplementedError