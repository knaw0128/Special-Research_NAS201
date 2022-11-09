from nats_bench import create
import os
import numpy as np
from pprint import pprint
import pickle
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--torch_home",
        type=str,
        default=os.getcwd(),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Choosing which dataset to use",
        default='cifar10-valid',
        choices=['cifar10-valid', 'cifar10', 'cifar100', 'ImageNet16-120'],
    )
    args = parser.parse_args()
    return args

def main(args):
    os.environ['TORCH_HOME'] = args.torch_home

    output_dir=os.path.join(args.torch_home,'NASBENCH_201_dict')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    

    # Create the API instance for the topology search space in NATS
    api = create(None, 'tss', fast_mode=True, verbose=False)
    template_array = np.zeros((8,8),dtype=int)
    template_array[0][1]=template_array[0][2]=template_array[0][4]=1
    template_array[1][3]=template_array[1][5]=1
    template_array[2][6]=template_array[3][6]=1
    template_array[4][7]=template_array[5][7]=template_array[6][7]=1
    final = []

    for i in range(len(api)):
        print('start model NO. {}'.format(i))
        record=dict()
        arch = api.query_meta_info_by_index(i)
        total_train_epo = arch.get_total_epoch(args.dataset) # 12 for cifar10 training 

        train_met = arch.get_metrics(args.dataset,'train')
        record['train_accuracy'] = train_met['accuracy']
        record['train_loss'] = train_met['loss']
        record['train_time'] = train_met['cur_time']
        record['train_accu_time'] = train_met['all_time']

        if(args.dataset != 'cifar10'):
            valid_met = arch.get_metrics(args.dataset,'x-valid')
            record['valid_accuracy'] = valid_met['accuracy']
            record['valid_loss'] = valid_met['loss']
            record['valid_time'] = valid_met['cur_time']
            record['valid_accu_time'] = valid_met['all_time']
        else:
            record['valid_accuracy'] = 0
            record['valid_loss'] = 0
            record['valid_time'] = 0
            record['valid_accu_time'] = 0

        test_met = arch.get_metrics(args.dataset,'ori-test')
        record['test_accuracy'] = test_met['accuracy']
        record['test_loss'] = test_met['loss']
        record['test_time'] = test_met['cur_time']
        record['test_accu_time'] = test_met['all_time']

        if(args.dataset in ['cifar100', 'ImageNet16-120']):
            xtest_met = arch.get_metrics(args.dataset,'x-test')
            record['x-test_accuracy'] = xtest_met['accuracy']
            record['x-test_loss'] = xtest_met['loss']
            record['x-test_time'] = xtest_met['cur_time']
            record['x-test_accu_time'] = xtest_met['all_time']
        
        arch_str = api.query_info_str_by_arch(i).split('\n')[0]
        tmp_list = api.str2lists(arch_str)
        # nodes: 0->1  0->2  1->2  0->3  1->3  2->3
        # layer:    1   2       3      4    5   6    
        # 0-th-op is 'none', 1-th-op is 'skip_connect',2-th-op is 'nor_conv_1x1'
        # 3-th-op is 'nor_conv_3x3', 4-th-op is 'avg_pool_3x3'    
        arch_list=[]
        arch_list.append('INPUT')
        for j in range(3):
            for k in range(j+1):
                arch_list.append(tmp_list[j][k][0])
        arch_list.append('OUTPUT')
        now_array = template_array.copy()
        # for i,j in zip(arch_list,range(len(arch_list))):
        #     if i == 'none':
        #         for k in range(8):
        #             now_array[j+1][k]=0
        final.append([now_array,arch_list,record])

    file_path=os.path.join(output_dir, args.dataset+'_model_label.pkl')
    with open(file_path,'wb') as file:
        pickle.dump(final, file)

if __name__ == '__main__':
    args = parse_args()
    main(args)


