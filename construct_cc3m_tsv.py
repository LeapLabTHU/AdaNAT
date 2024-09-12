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
process_json_files('/path/to/your', 'cc3m_val_val.tsv')