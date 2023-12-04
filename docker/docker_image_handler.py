import subprocess
import importlib
import time
import os
import random
import argparse
from typing import Union
from collections import defaultdict
import glob
import abc

# ----------------------------------------------------------------

try:
    import docker
except ModuleNotFoundError:
    install_module("docker")
    import docker

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress, 
        TextColumn, 
        BarColumn, 
        TaskProgressColumn, 
        TimeRemainingColumn, 
        TimeElapsedColumn, 
        FileSizeColumn,
        SpinnerColumn
    )
except ModuleNotFoundError:
    install_module("rich")
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress, 
        TextColumn, 
        BarColumn, 
        TaskProgressColumn, 
        TimeRemainingColumn, 
        TimeElapsedColumn, 
        FileSizeColumn,
        SpinnerColumn
    )

# ----------------------------------------------------------------

def install_module(module_name):
    try:
        subprocess.check_call(["pip", "install", module_name])
        print(f"Successfully installed {module_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {module_name}")

def import_module(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"Imported {module_name} successfully")

        if module:
            globals()[module_name] = module

    except ImportError:
        print(f"Failed to import {module_name}")

# ----------------------------------------------------------------

class DockerFinder:

    def __init__(self):
        self._images = defaultdict(dict)
        self._max_index = 0

    def get_all(self):
        return self._images

    def _is_name_includes(self, repository:str, includes: list):
        if not includes: return True
        for inc in includes:
            if inc not in repository:
                return False
        return True

    def _is_name_excludes(self, repository:str, excludes: list):
        if not excludes: return True
        for exc in excludes:
            if exc in repository:
                return False
        return True

    def find(self, includes: list=[], excludes: list=[]):
        client = docker.from_env()
        images = client.images.list()

        for image in images:
            
            if not image.attrs["RepoTags"]: continue

            repo_tag = image.attrs["RepoTags"][0]

            if not (self._is_name_excludes(repo_tag, excludes) \
                and self._is_name_includes(repo_tag, includes)):
                    continue
            
            self._images[self._max_index].update({
                "id": image.id,
                "repo": repo_tag,
                "created": image.attrs["Created"],
                "size": image.attrs["Size"],
            })
            self._max_index += 1

        return self._images

class DockerSaver:

    def save1(self, image_name:str, output_file:str):
        return subprocess.Popen(["docker", "save", "-o", output_file, image_name])

    def save(self, image_name, output_file):
        return subprocess.Popen(["docker", "save", "-o", output_file, image_name])

    def __call__(self, image_name, output_file):
        return self.save(image_name, output_file)

class DockerLoader:

    def load(self, input_file: str):
        if not os.path.exists(input_file):
            raise FileNotFoundError('The tarball file not found. ({})'.format(input_file))
        return subprocess.Popen(f"docker load -i {input_file} > /dev/null", shell=True)

    def __call__(self, input_file: str):
        return self.load(input_file)

# ----------------------------------------------------------------

class RichTable:

    def __init__(self):
        self.console = Console()
        self.table = Table(title="List of Inno Docker")
        self.headers = []

    def define_header(self, headers: list):
        self.headers = headers
        for header in headers:
            self.table.add_column(header)
    
    def update(self, values: tuple):
        self.table.add_row(*values)

    def print_out(self):
        # self.console.clear()
        self.console.print(self.table)

# ----------------------------------------------------------------

def convert_bytes(
    bytes: int, 
    unit: str, 
    num_decimal: int = 3, 
    need_text: bool = False ) -> Union[str, int]:
    
    unit_map = { "KB": 1, "MB": 2, "GB": 3, "TB": 4 }
    
    N = unit_map.get(unit)
    if N is None:
        raise TypeError("Expect unit is {}, but got {}".format(
            ', '.join(map.keys()), unit ))
    
    ret_value = round(bytes/(1024**N), num_decimal)

    return f"{ret_value} {unit}" if need_text else ret_value

class InnoDocker(abc.ABC):

    _save_folder = ""
    _processes = defaultdict()
    progress = None
    table = None

    def __init__(self, folder: str):
        self.save_folder = folder
        # Define rich parameters
        self.progress = self.get_rich_progress()
        self.table = RichTable()

    @property
    def save_folder(self):
        return self._save_folder
    
    @save_folder.setter
    def save_folder(self, folder_path:str):
        if not isinstance(folder_path, str):
            raise TypeError("Should be a string.")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created a folder for saving docker image: {folder_path}")
        self._save_folder = folder_path

    def get_rich_progress(self):
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn() )

    @abc.abstractmethod
    def define_table(self):
        pass

    @abc.abstractmethod
    def update_table(self):
        pass

    @abc.abstractmethod
    def start(self):
        pass

class InnoDockerSaver(InnoDocker):
    
    def __init__(self, folder: str):
        super().__init__(folder)
        
        self.saver = DockerSaver()
        self.finder = DockerFinder()
        self.finder.find(includes=['inno'], excludes=['none'])
        self.data = self.finder.get_all().items()
        self.define_table()
        self.update_table(self.data)

    def define_table(self):
        # Define rich parameters
        self.table.define_header(["Index", "Name", "Size", "Create"])

    def update_table(self, data: dict):
        for idx, docker_data in data.items():
            self.table.update((
                str(idx), 
                docker_data['repo'], 
                str(convert_bytes(docker_data['size'], "MB", need_text=True)),
                str(docker_data['created'])
            ))
        self.table.print_out()

    def start(self):
        
        # Temp process for rich progress
        rich_process = defaultdict() 

        # Save
        for idx, image in self.finder.get_all().items():
            
            image_name = image["repo"]
            base_name = os.path.basename(image_name).strip()
            output_file = os.path.join(
                self._save_folder,
                f"{base_name}.tar" )

            self._processes[idx] = {
                "id": base_name,
                "proc": self.saver(image_name, output_file),
                "input": image_name,
                "output": output_file }

            # Add Task
            rich_process[idx] = self.progress.add_task(base_name, total=100)

        with self.progress:
        
            # Wait
            while not self.progress.finished:
                for idx, data in enumerate(self._processes.values()):
                    process_running = (data['proc'].poll() is None)
                    file_exists = (os.path.exists(data['output']))
                    if not process_running and file_exists:
                        self.progress.update(rich_process[idx], completed=100, style="green")
                        continue
                    self.progress.update(rich_process[idx], advance=random.uniform(0.01, 0.05))
                    
                    # self.progress.update(rich_process[idx], advance=random.uniform(0.05, 1), style="green")

                time.sleep(0.5)

    def __del__(self):
        for key, data in self._processes.items():
            data["proc"].terminate()

class InnoDockerLoader(InnoDocker):
    
    _tars = None

    def __init__(self, folder: str):
        super().__init__(folder)
        self.loader = DockerLoader()

        # Find tar
        self._tars = self.find_tars( self.save_folder )
        
        # Rich
        self.define_table()
        self.update_table()

    def find_tars(self, folder:str, extension: str="tar") -> dict:        
        temp = defaultdict(dict)
        for idx, tar in enumerate(glob.glob(os.path.join(folder, f"*.{extension}"))):
            size = convert_bytes(os.stat(tar).st_size, unit='MB', need_text=True)
            base_name = os.path.basename(tar)
            temp[idx] = {
                'input': tar,
                'base_name': base_name,
                'size': size }
        print(f'Find {len(temp)} tar files')
        return temp

    def define_table(self):
        self.table.define_header(["Index", "Name", "Size"])

    def update_table(self):
        
        for idx, data in self._tars.items():
            self.table.update((
                str(idx),
                data['input'],
                data['size']
            ))
        self.table.print_out()

    def start(self):

        # Temp process for rich progress
        rich_process = defaultdict()

        for idx, data in self._tars.items():
            
            self._processes[idx] = {
                "id": data["base_name"],
                "proc": self.loader(os.path.realpath(data["input"])),
                "input": data["input"],
            }
            rich_process[idx] = self.progress.add_task(
                data["base_name"], total=100)

        with self.progress:
            # Wait
            while not self.progress.finished:

                for idx, data in enumerate(self._processes.values()):
                    process_running = (data['proc'].poll() is None)
                    if not process_running:
                        self.progress.update(rich_process[idx], completed=100, style="green")
                        continue
                    self.progress.update(rich_process[idx], advance=random.uniform(0.01, 0.05))
                    
                    # self.progress.update(rich_process[idx], advance=random.uniform(0.05, 1), style="green")

                time.sleep(0.5)
            
    def __del__(self):
        for key, data in self._processes.items():
            data["proc"].terminate()

# ----------------------------------------------------------------

def save_docker_image(args):
    dif = InnoDockerSaver(args.folder)
    dif.start()

def load_docker_image(args):
    dif = InnoDockerLoader(args.folder)
    dif.start()

# ----------------------------------------------------------------

EXEC = {
    "load": load_docker_image,
    "save": save_docker_image
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', required=True, choices=list(EXEC.keys()))
    parser.add_argument('-f', '--folder', required=True, help="the folder includes tarball.")
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    EXEC[args.mode](args)