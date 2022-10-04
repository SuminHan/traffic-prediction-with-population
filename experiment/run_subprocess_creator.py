models = ['MyDCGRUSTE', 'MyGMSTARK', 'MyDCGRUSTE', 'MyGMAN']
regions = ['gangnam', 'hongik', 'jamsil']
with open('run_subprocess_list.sh', 'w') as fp:
    for r in regions:
        for m in models:
            line = f'python trainmodel.py --model_name={m} --region={r}'
            fp.write(line + '\n')