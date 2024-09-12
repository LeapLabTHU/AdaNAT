# CC3M Training Dataset Preparation

## Download raw data in caption-url pair
Please download `Training split` via [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download)

## Download raw train data
ref: [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md)

### Install img2dataset
``` shell
pip install img2dataset
```

### Add head to .tsv files
``` shell
apt install sed

sed -i '1s/^/caption\turl\n/' Train_GCC-training.tsv
```

### Download image data
ref : [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT/blob/main/data/T-X_pair_data/cc3m/prepare.md)
``` shell
# Make a dir
mkdir cc3m

# Download training image
img2dataset --url_list Train_GCC-training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder cc3m/training --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True
```
**Note that: **
- `url_list` A file with the list of url of images to download. It can be a folder of such files. (required)
- `image_size` The size to resize image to (default 256)
- `output_folder` The path to the output folder. (default "images")
- `processes_count` The number of processes used for downloading the pictures. This is important to be high for performance. (default 1)
- `thread_count` The number of threads used for downloading the pictures. This is important to be high for performance. (default 256)
- `output_format` decides how to save pictures (default files)
  - `files saves` as a set of subfolder containing pictures
  - `webdataset` saves as tars containing pictures
  - ...
- `url_col` the name of the url column for parquet and csv (default url)
- `caption_col` the name of the caption column for parquet and csv (default None)
- `enable_wandb` whether to enable wandb logging (default False)

### decompress all data
Define a shell script
``` shell
vi untar.sh
```
type in,
``` shell
for file in *.tar; do
	tar -xvf "$file" 
done
```
give executive permission,
``` shell
chmod 777 untar.sh
```
decompress training image dataset,
``` shell
cp untar.sh cc3m/training
cd cc3m/training
./untar.sh
```

## Generate new .tsv files
### Create a Python script
```shell
vi gen_train_tsv.py
```
, type in :
```python
import json
import os
from io import BytesIO
from PIL import Image
import base64

from tqdm import tqdm

def process_json_files(directory, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    idx = 0
    for filename in tqdm(os.listdir(directory),desc="Parsing dataset..."):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    continue

                if data.get('status') == 'success':
                    caption = data.get('caption', '')
                    key = data.get('key', '') + '.jpg'

                    buffered = BytesIO()
                    Image.open(key).save(buffered, format="JPEG")
                    img_bytes = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    with open(output_file, 'a') as out_file:
                        out_file.write(f"{key}\t{idx}\t{caption}\t{img_bytes}\n")
                        idx += 1


# 调用函数，指定目录和输出文件的路径
# Call the function, specifying the directory and output file paths
process_json_files('path/to/your/train/json/files', 'cc3m_train.tsv')
```
, run the Python file:
``` shell
python gen_train_tsv.py
```

After this, please move the `cc3m_train.tsv` file to the `assets` folder.