import json
from glob import glob
import os

import argparse
parser = argparse.ArgumentParser(
                    prog='make_manifest',
                    description='Create a JSON file with locations and metadata of NR sims')
parser.add_argument('--dir', type=str)
args = parser.parse_args()

omega = {}
with open('Models.txt') as ff:
    for line in ff.readlines():
        tag = line.split(' ')[0]
        omega[tag] = float(line.split('=')[-1])

searchglob = os.path.join(args.dir, 'PSI4_nophase/*/*l2_m2*.asc')
        
wflst = {'group': 'Nico_HeadOn'}
for line in sorted(glob(searchglob)):
    record = {}
    path, fname = os.path.split(line)
    simtag = fname.split('_l2')[0]
    record = {'path': path, 'simtag': simtag} # 'filename': fname,
    if 'reco' in simtag:
        tag1, tag2 = simtag.split('_')[:2]
        tag = '_'.join([tag1, tag2])
        w1 = omega[tag1.lstrip('r')]
        w2 = omega[tag2]
    else:
        tag = simtag.split('_')[0]
        w1 = w2 = omega[tag]
    record['metadata'] = {'omega1': w1, 'omega2': w2}
    
    wflst[tag] = record

json.dump(wflst, open('manifest_Nico_HeadOn.json', 'w'))
