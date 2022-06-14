import argparse
import numpy as np
import pandas as pd
import os
import shutil


def create_folders(path, len):
    for i in range(len):
        os.mkdir(path + str(i))


# Function for move data from one folder to another
def split_data(actual_dir, train_dir, teste_dir, random_list, split_size=[0.7, 0.3]):
    for i in os.listdir(actual_dir):
        path = actual_dir + "/" + i
        path_size = len(os.listdir(path))
        # Moving data from actual_dir to train_dir
        count = 0
        for k in range(path_size):
            if count <= int(path_size * (split_size[0])):
                shutil.move(
                    os.path.join(path, os.listdir(path)[int(random_list[k])]),
                    os.path.join(train_dir, i),
                )
                count += 1
            # Moving data from actual_dir to test_dir
            elif count <= int(path_size * (split_size[0] + split_size[1])):
                shutil.move(
                    os.path.join(path, os.listdir(path)[int(random_list[k])]),
                    os.path.join(teste_dir, str(i)),
                )
                count += 1


def make_dataset(args):
    data_root = args.data_root
    random_list_file = args.random_list_file

    # Configuring the folders structure
    classes = {
        0: "ViseScrew",
        1: "Clamp",
        2: "Nut",
        3: "Base",
        4: "All",
        5: "HeadPlate",
        6: "BallJoint",
        7: "Screw",
        8: "ClampScrew",
        9: "Slider",
    }
    initial_path = data_root.split("/")
    initial_path = "/".join(initial_path[:-1])

    if os.path.exists(data_root):

        for i in os.listdir(data_root):
            for index, classe in classes.items():
                if classe == i:
                    os.rename(
                        os.path.join(data_root, i), os.path.join(data_root, str(index))
                    )

        train_dir = initial_path + "/train/"
        test_dir = initial_path + "/test/"

        os.makedirs(train_dir)
        os.makedirs(test_dir)

        create_folders(train_dir, len(os.listdir(data_root)))
        create_folders(test_dir, len(os.listdir(data_root)))

        # A random list for splitting the data in a aleatory random way
        random_list = (
            pd.read_csv(random_list_file, sep=";", header=None).iloc[0].tolist()
        )

        split_data(data_root, train_dir, test_dir, random_list)
        shutil.rmtree(data_root)

        print("Dataset successfully created")

    else:
        print("The initial configuration is already done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make dataset")
    parser.add_argument(
        "--data_root", type=str, default=None, help="Path to the data root"
    )
    parser.add_argument("--random_list_file", type=str, default="random_list.csv")
    args = parser.parse_args()

    make_dataset(args)
