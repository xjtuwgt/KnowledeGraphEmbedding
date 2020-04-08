import numpy as np
import os
import pandas as pd
import seaborn as sns

path = '../logs/'
parameter_names = ['hidden_dim', 'ent_embed_dim', 'rel_embed_dim', 'embed_dim',
                   'gamma', 'batch_size', 'learning_rate', 'adam_weight_decay', 'att_drop', 'input_drop',
                   'fea_drop', 'top_k', 'hops', 'layers', 'alpha', 'slope', 'loss_type', 'neg_on']

model_data_names = ['model', 'data_path']
SPLIT_FLAG = 'INFO'
PARAMETER_FLAG = '='
TRAIN_LOSS_FLASS = 'Training average loss'
MRR_FLAG = 'Valid MRR'


def parameter_extractor(line):
    resultline = line.split(SPLIT_FLAG)[-1].strip()
    results = resultline.split(PARAMETER_FLAG)
    if len(results) >= 2:
        key, value = results[0].strip(), results[1].strip()
    else:
        return None, None
    if key in model_data_names:
        return key, value
    if key in parameter_names:
        return key, float(value)
    return None, None

def deep_log_analysis(file_name):
    parameter_dict = {}
    train_loss_list = []
    valid_mrr_list = []
    best_mrr = 0
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for x in lines:
            if PARAMETER_FLAG in x:
                key, value = parameter_extractor(x)
                if key is not None:
                    parameter_dict[key] = value
            if TRAIN_LOSS_FLASS in x:
                resultline = x.split(SPLIT_FLAG)[-1].split(':')
                loss = float(resultline[-1].strip())
                train_loss_list.append(loss)

            if MRR_FLAG in x:
                resultline = x.split(SPLIT_FLAG)[-1].split(':')
                mrr = float(resultline[-1].strip())
                valid_mrr_list.append(mrr)
    if len(valid_mrr_list) > 0:
        valid_mrr_array = np.array(valid_mrr_list)
        best_mrr = valid_mrr_array.max()
    parameter_dict['best_mrr'] = best_mrr
    return parameter_dict


def best_config_analysis(log_folder, dataname):
    log_files = [x for x in os.listdir(log_folder) if x.endswith(".log")]
    max_mrr = 0
    best_config = ''
    parameter_mrr_list = []
    for idx, file_name in enumerate(log_files):
        if dataname in file_name:
            mrr_parameter = deep_log_analysis(log_folder + file_name)
            parameter_mrr_list.append(mrr_parameter)
            if max_mrr < mrr_parameter['best_mrr']:
                max_mrr = mrr_parameter['best_mrr']
                best_config = file_name
    return max_mrr, best_config, parameter_mrr_list

def parameter_analysis(parameter_mrr_list):
    dataframe = pd.DataFrame(parameter_mrr_list)
    dataframe = dataframe[dataframe['best_mrr']>0.005]
    filter_names = []
    for col in dataframe.columns:
        if (dataframe[col].unique().size == 1):
            filter_names.append(col)
    dataframe = dataframe.drop(columns=filter_names)
    for col in dataframe.columns:
        if col != 'best_mrr':
            print('-------------{}vs{}-------'.format(col, 'mean mrr'))
            print(dataframe.groupby(col)['best_mrr'].mean())
            print('=======================\n\n')

if __name__ == '__main__':
    max_mrr, config, para_mrr_list = best_config_analysis(path, 'wn18rr')
    parameter_analysis(para_mrr_list)

    print('Best MRR {} config {}'.format(max_mrr, config))

    # max_mrr, config, para_mrr_list = best_config_analysis(path, 'FB15k')
    # print('Best MRR {} config {}'.format(max_mrr, config))
    # parameter_analysis(para_mrr_list)