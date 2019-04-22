import os

for i in range(1, 7):
    name = './data/heads_and_tails_' + str(i) + '/categories_samples/'
    for category_folder in os.listdir(name):
        category_source_path = name + category_folder + '/'
        category_dest_path = './data/test_dataset/categories_samples/' + category_folder + '/'
        if not os.path.isdir(category_dest_path):
            os.mkdir(category_dest_path)
        for subcat_folder in os.listdir(category_source_path):
            subcat_source_path = category_source_path + subcat_folder + '/'
            subcat_dest_path = category_dest_path + subcat_folder + '/'
            if not os.path.isdir(subcat_dest_path):
                os.mkdir(subcat_dest_path)
            for sample in os.listdir(subcat_source_path):
                source_sample_path = subcat_source_path + sample
                num = len([name for name in os.listdir(subcat_dest_path) if os.path.isfile(os.path.join(subcat_dest_path, name))])
                dest_sample_path = subcat_dest_path + str(num) + '.wav'
                os.popen('cp' + source_sample_path + ' ' + dest_sample_path)
