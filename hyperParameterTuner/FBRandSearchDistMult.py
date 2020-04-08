import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from hyperParameterTuner.randomsearch import KGERandomSearchJob
from kgembedUtils.ioutils import remove_all_files

def HypeParameterSpace(model_name):
    learning_rate = {'name': 'learning_rate', 'type': 'range', 'bounds': [0.0005, 0.01], "log_scale": True}
    smooth_factor = {'name': 'gamma', 'type': 'range', 'bounds': [0.05, 0.5]}
    feat_drop = {'name': 'fea_drop', 'type': 'range', 'bounds': [0.1, 0.6]}
    input_drop = {'name': 'inp_drop', 'type': 'range', 'bounds': [0.1, 0.6]}
    edge_drop = {'name': 'edge_drop', 'type': 'range', 'bounds': [0.1, 0.5]}
    att_drop = {'name': 'att_drop', 'type': 'range', 'bounds': [0.2, 0.6]}
    batch_size = {'name': 'batch_size', 'type': 'choice', 'values': [512, 1024]}
    hidden_dim = {'name': 'hidden_dim', 'type': 'fixed', 'value': 256}
    ent_embed_dim = {'name': 'ent_embed_dim', 'type': 'fixed', 'value': 200}
    num_heads = {'name': 'num_heads', 'type': 'choice', 'values': [8, 16]}
    rel_embed_dim = {'name': 'rel_embed_dim', 'type': 'fixed', 'value': 200}
    project_on = {'name': 'project_on', 'type': 'fixed', 'value': 0}
    negative_on = {'name': 'negative_on', 'type': 'fixed', 'value': 0}
    zero_on = {'name': 'zero_on', 'type': 'fixed', 'value': 1}
    loss_type = {'name': 'loss_type', 'type': 'fixed', 'value': 1}
    embed_dim = {'name': 'embed_dim', 'type': 'fixed', 'value': 256}
    top_k = {'name': 'top_k', 'type': 'choice', 'values': [2, 3, 4, 5, 10]}
    max_steps = {'name': 'max_steps', 'type': 'fixed', 'value': 80000}
    layers = {'name': 'layers', 'type': 'choice', 'values': [1, 2]}
    alpha = {'name': 'alpha', 'type': 'range', 'bounds': [0.2, 0.8]}
    hops = {'name': 'hops', 'type': 'choice', 'values': [2, 3, 4, 5]}
    feed_forward = {'name': 'feed_forward', 'type': 'fixed', 'value': 0}
    topk_type = {'name': 'topk_type', 'type': 'fixed', 'value': 'local'}
    adam_weight_decay = {'name': 'adam_weight_decay', 'type': 'fixed', 'value': -1}
    conv_embed_shape1 = {'name': 'conv_embed_shape1', 'type': 'fixed', 'value': -1}
    conv_channels = {'name': 'conv_channels', 'type': 'choice', 'values': [100, 200]}
    conv_filter_size = {'name': 'conv_filter_size', 'type': 'fixed', 'value': 5}
    model_name = {'name': 'model_name', 'type': 'fixed', 'value': model_name}
    #++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, adam_weight_decay, att_drop, negative_on, feat_drop, input_drop, ent_embed_dim, rel_embed_dim, conv_embed_shape1,
                    conv_channels, topk_type, feed_forward, edge_drop, zero_on,
                       max_steps, top_k, hops, num_heads, layers, alpha, loss_type, hidden_dim, embed_dim, smooth_factor, model_name,conv_filter_size,
                       batch_size, project_on]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space


def generate_random_search_bash(model_name, data_name, task_num, bash_save_path='../fb_jobs/'):
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    if bash_save_path and not os.path.exists(bash_save_path):
        os.makedirs(bash_save_path)
    search_space = HypeParameterSpace(model_name=model_name)
    random_search_job = KGERandomSearchJob(data_name=data_name,
                                           search_space=search_space, graph_on=1, mask_on=1)
    for i in range(task_num):
        task_id, parameter_id = random_search_job.single_task_trial(i+1)
        with open(bash_save_path + 'run_' + task_id +'.sh', 'w') as rsh_i:
            command_i = 'bash run.sh ' + parameter_id
            rsh_i.write(command_i)
    print('{} jobs are generated in {}'.format(task_num, bash_save_path))

if __name__ == '__main__':
    generate_random_search_bash(model_name='ConvE', data_name='FB15k-237', task_num=40)