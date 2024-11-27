#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:00:14 2024

@author: erri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:08:41 2024

@author: erri
"""

import os

def search_string_in_files(folder_path, search_string):
    # Iterate through all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):  # Check if the file is a Python file
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if search_string in content:
                            print(f"Found in file: {file_path}")
                except UnicodeDecodeError:
                    print(f"Skipping file due to encoding issue: {file_path}")

# Replace 'your_folder_path' with the path of the folder you want to search in
folder_path = os.getcwd()

# Replace 'your_search_string' with the string you want to search for
search_string = 'report_all_modes.txt'

search_string_in_files(folder_path, search_string)
