import os

for i in range(1, 7):
    name = './data/heads_and_tails_' + str(i) + '/categories_samples/'
    for folder in os.listdir(name):
        path_to_file = name + folder
        for sample in os.listdir(path_to_file):
            sample_path = path_to_file + sample
            print(sample_path)
