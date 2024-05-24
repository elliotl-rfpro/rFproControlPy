import os
import json
from typing import List

exclude = ['graph.png', 'names.txt']


def list_simulations(directory: str) -> List[str]:
    # Finds all simulation folders in a given directory and saves in a 'names.txt' file
    folder_names = os.listdir(directory)
    clean_names = []
    for i in range(len(folder_names)):
        if folder_names[i] not in exclude:
            clean_names.append(folder_names[i])
    with open(f'{directory}/names.txt', 'w') as file:
        file.write(json.dumps(clean_names))

    return folder_names
