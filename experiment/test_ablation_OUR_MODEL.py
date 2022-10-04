import time
import os
import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

print_cols = ['Model', 'cell', 'cnn', 'BI', 'SM', 'TM', 'DY', 'POI', 'SAT', 'LTE']
region_list = ['gangnam', 'hongik', 'jamsil']

opt_keys = ['region', 'module', 'cell_size', 'cnn_size', 
    'NOBIM', 'SMASK', 'TMASK', 'NODYC', 'NOPOI', 'NOSAT', 'NOLTE']
def load_opt(firstline):
    args = dotdict()
    for token in firstline.split(','):
        k, v = token.strip().split('=')
        if v == 'True':
            v = True
        elif v == 'False':
            v = False

        if k == 'region':
            v = v[1:-1]

        args[k] = v

    record = [args.region, 'OURS_0', args.cell_size, args.cnn_size]
        
    if args.NOBIM:
        record.append('X')
    else:
        record.append('')

    if args.SMASK:
        record.append('O')
    else:
        record.append('')

    if args.TMASK:
        record.append('O')
    else:
        record.append('')
        
    if args.NODYC:
        record.append('X')
    else:
        record.append(f'{args.lambda_value}')
        
    if args.NOPOI:
        record.append('X')
    else:
        record.append('')
        
    if args.NOSAT:
        record.append('X')
    else:
        record.append('')
        
    if args.NOLTE:
        record.append('X')
    else:
        record.append('')

    rdict = dict()
    rdict = {k:v for k, v in zip(opt_keys, record)}
    
    return rdict


def load_tested():
    flist = sorted([fname for fname in os.listdir('data') if 'train' in fname])
    region_dict = {}
    opt_dict = {}
    # print('\t'.join(cols))
    for fname in flist:
        if not '12-3' in fname:
            continue
        with open(os.path.join('data', fname)) as fp:
            all_lines = fp.read().split('\n')
            firstline = all_lines[0]
            lines = all_lines[-30:]
        
        all_lines = '\n'.join(all_lines)
        if not (('performance in each prediction step' in all_lines) and ('**** testing model ****' in all_lines)):
            continue
        
        test_opt = load_opt(firstline)
        records = '\t'.join(test_opt[k] for k in opt_keys)
        opt_dict['\t'.join(records)] = test_opt

        
        lines = all_lines[all_lines.index('performance in each prediction step'):].split('\n')
        row = []
        for i in [-6, -5, -4, -3]:
            row.extend(lines[i].split()[-3:])

        key = '\t'.join(test_opt[k] for k in opt_keys[1:])
        region = test_opt['region']
        val = '{:.3f}'.format(np.mean([float(r) for r in [row[-6], row[-9], row[-12]]]))
        region_dict.setdefault(key, {})
        region_dict[key][region] = val

    return region_dict, opt_dict

def print_tested():
    lines = []
    region_dict, opt_dict = load_tested()

    lines.append('\t'.join(print_cols) + '\t' + '\t'.join(region_list))
    vals = []
    for key in sorted(region_dict.keys()):
        # print(key)
        row = []
        val = []
        for reg in region_list:
            if reg in region_dict[key]:
                row.append(region_dict[key][reg])
                val.append(float(region_dict[key][reg]))
            else:
                row.append('-')
                val.append(np.inf)
                
        lines.append(key + '\t' + '\t'.join(row))
        vals.append(val)

    vals = np.min(vals, 0)
    lines.append('\t'.join(['' for _ in range(len(print_cols)-1)]) + '\t' + 'MIN:' + '\t' + '\t'.join([str(f) for f in vals]))

    return '\n'.join(lines)

if __name__ == '__main__':
    print(print_tested())
