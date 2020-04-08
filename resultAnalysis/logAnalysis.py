path = '../logs/'
# path = '../with_loop_logs/logs_fb_0.362/'
import os
SPLIT_FLAG = 'INFO'
mrr_flag = 'MRR at step'
def log_analysis(file_name):
    best_mrr = 0.0
    with open(file_name, 'r') as f:
        x = f.readlines()
        for line in x:
            if mrr_flag in line:
                resultline = line.split(SPLIT_FLAG)[-1].split(':')
                mrr = float(resultline[-1].strip())
                if mrr > best_mrr:
                    best_mrr = mrr
    if 'FB' in file_name:
        print('{}\t{}\t FB'.format(file_name, best_mrr))
    else:
        print('{}\t{}\t WN'.format(file_name, best_mrr))
    return best_mrr

def best_config_analysis(log_folder, dataname):
    log_files = [x for x in os.listdir(log_folder) if x.endswith(".log")]
    max_mrr = 0
    best_config = ''
    for idx, file_name in enumerate(log_files):
        if dataname in file_name:
            mrr = log_analysis(log_folder + file_name)
            if max_mrr < mrr:
                max_mrr = mrr
                best_config = file_name
    return max_mrr, best_config


if __name__ == '__main__':
    max_mrr, config = best_config_analysis(path, 'wn18rr')
    print('Best MRR {} config {}'.format(max_mrr, config))

    max_mrr, config = best_config_analysis(path, 'FB15k')
    print('Best MRR {} config {}'.format(max_mrr, config))