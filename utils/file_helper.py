import os
import zipfile
import logging as log

def compress_files(zip_path: str, zip_files: list, keep_dir: bool=True) -> bool:
    log.debug(f"ZIP {len(zip_files)} Files -> {zip_path} {'(keep dir)' if keep_dir else ''}")
    with zipfile.ZipFile(zip_path, mode='w') as zf:
        for _file in zip_files:
            _file = os.path.relpath(_file)
            log.debug('\t- Compress file: {}'.format(_file))
            if keep_dir:
                zf.write(_file)
            else:
                zf.write( _file, os.path.join(
                    '.', os.path.basename(_file)))

    # Double check
    return os.path.exists(zip_path)

def is_zip(file_path: str) -> bool:
    return zipfile.is_zipfile(file_path)


def valid_zip(file_path: str):
    """Check the ZIP file is available

    Args:
        file_path (str): path to zip file

    Raises:
        FileNotFoundError: file not found
        TypeError: not zip file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Zip file not found: {file_path}')

    if zipfile.is_zipfile(file_path):
        return
        
    raise TypeError(f"The file is not a zip: {file_path}")

def extract_files(zip_path: str, trg_folder: str) -> bool:

    valid_zip(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(trg_folder)
    except Exception as e:
        raise e

    log.debug(f"UNZIP {zip_path} -> {len(os.listdir(trg_folder))} Files")
    return os.path.exists(trg_folder)