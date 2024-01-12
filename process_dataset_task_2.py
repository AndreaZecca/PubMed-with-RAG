import pandas as pd 
import json

def process_medmcqa(dataset_path: str) -> None:
    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        print(f"Error reading dataset, {e}")

    # We only need the context
    context_df = df["context"]

    # Context is a dictionary with 'contexts', 'labels' and 'meshes'. get only 'contexts'
    context_df = context_df.apply(lambda x: x['contexts'])

    # Save context to file as json
    context_df.to_json("datasets/context/medmcqa/context.json", orient="records")


def process_textbooks_dataset(textbooks_folder_path: str) -> None:

    # Get all .txt files in the folder
    import glob
    import os

    files = glob.glob(os.path.join(textbooks_folder_path, "*.txt"))

    # Read all files and save to a single file 
    # Create the file if it doesn't exist
    with open(f"datasets/context/all_books.txt", "w+") as f:
        for file in files:
            print(f"Reading file {file}")
            with open(file, "r") as f2:
                f.write(f2.read())


#medmcqa_dataset_path: str = "datasets/context/medmcqa/train-00000-of-00001.parquet"
#process_medmcqa(medmcqa_dataset_path)

#textbooks_folder_path: str = "datasets/context/medqa/"
#process_textbooks_dataset(textbooks_folder_path)
