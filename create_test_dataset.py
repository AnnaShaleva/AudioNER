import os
import sys
def get_samples_from_path(name, dataset_name):
    for category_folder in os.listdir(name):
        category_source_path = name + category_folder + '/'
        category_dest_path = './data/' + dataset_name + '/categories_samples/' + category_folder + '/'
        if not os.path.isdir(category_dest_path):
            os.mkdir(category_dest_path)
        for subcat_folder in os.listdir(category_source_path):
            subcat_source_path = category_source_path + subcat_folder + '/'
            subcat_dest_path = category_dest_path + subcat_folder + '/'
            if not os.path.isdir(subcat_dest_path):
                os.mkdir(subcat_dest_path)
            for sample in os.listdir(subcat_source_path):
                source_sample_path = subcat_source_path + sample
                #num = len([name for name in os.listdir(subcat_dest_path) if os.path.isfile(os.path.join(subcat_dest_path, name))])
                dest_sample_path = subcat_dest_path + sample
                os.popen('cp ' + source_sample_path + ' ' + dest_sample_path)

if __name__=='__main__':
    dataset_name = sys.argv[1]
    for i in range(1, 13):
        name = './data/heads_and_tails_' + str(i) + '/categories_samples/'
        get_samples_from_path(name, dataset_name)
    names = ['./data/echo_of_moscow_dataset/categories_samples/',
    './data/snailkick_dataset/categories_samples/',
    './data/sk2_dataset/categories_samples/',
    './data/youtube_test/categories_samples/',
    './data/audiobooks_dataset/categories_samples/',
    './data/ht_old_dataset/categories_samples/',
    './data/stories_dataset/categories_samples/',
    './data/interviews_dataset/categories_samples/',
    './data/svoim_hodom_dataset/categories_samples/',
    './data/dud_dataset/categories_samples/']
    for name in names:
        get_samples_from_path(name, dataset_name)
