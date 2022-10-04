import os

cols = ['Region']
for i in [1, 2, 3, 'avg']:
    for c in ['mae', 'rmse', 'mape']:
        cols.append('{}_{}'.format(c,i))

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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

    record = [args.region, 'OURS']

    
    return record


flist = sorted([fname for fname in os.listdir('data') if 'train' in fname])
pregion = None
print('\t'.join(cols))
for fname in flist:
    if 'EXP' in fname:
        continue
    if not ('3-2_DY0.6_2021' in fname):
        continue

    with open(os.path.join('data', fname)) as fp:
        all_lines = fp.read().split('\n')
        firstline = all_lines[0]
        lines = all_lines[-10:]
    
    if len(lines) > 7 and lines[-7] == 'performance in each prediction step':
        row = []
        for i in [-6, -5, -4, -3]:
            row.extend(lines[i].split()[-3:])
        records = load_opt(firstline)
        if records[0] != pregion:
            pregion = records[0]
        print('\t'.join(records), '\t'.join(row), sep='\t')