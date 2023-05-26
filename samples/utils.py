# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import sys
import time
import zipfile
import gdown

try:
    from ..common import SERV_CONF
except:
    from common import SERV_CONF

def extract_file(zip_path:str, folder_name:str=None):
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError("ZIP Path is unavailable: {}".format(zip_path))
    
    if not folder_name:
        folder_name = os.path.splitext(zip_path)[0]

    print(zip_path, folder_name)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path=folder_name)

    os.remove(zip_path)

def download_file(file_path, file_url):
    check_data = file_path
    if os.path.splitext(file_path)[1] == '.zip':
        check_data = os.path.splitext(file_path)[0]

    if os.path.exists(check_data):
        return

    gdown.download(url=file_url, output=file_path, quiet=False, fuzzy=True)

def download_model(file_name, file_url):
    ext = '.zip'
    if not ext in file_name:
        file_name += ext
    file_path = os.path.join(SERV_CONF["MODEL_DIR"], file_name)
    download_file(file_path, file_url)
    extract_file(file_path)

def download_data(file_name, file_url):
    file_path = os.path.join(SERV_CONF["DATA_DIR"], file_name)
    download_file(file_path, file_url)