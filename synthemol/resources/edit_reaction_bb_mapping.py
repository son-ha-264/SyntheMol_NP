import pickle
import pandas as pd


PATH_PICKLE_OUT = "/home/sonh/MAINCE_project/SyntheMol_NP/synthemol/resources/real_small/reaction_to_building_blocks.pkl"
PATH_BB = "/home/sonh/MAINCE_project/SyntheMol_NP/synthemol/resources/real_small/building_blocks.csv"

PATH_PICKLE_IN = "/home/sonh/MAINCE_project/SyntheMol_NP/synthemol/resources/real/reaction_to_building_blocks.pkl"


df = pd.read_csv(PATH_BB)
set_bb = set(df['smiles'])

with open(PATH_PICKLE_IN, "rb") as f:
    a = pickle.load(f)

for key1 in a.keys():
    for key2 in a[key1].keys():
        a[key1][key2] = a[key1][key2].intersection(set_bb)

with open(PATH_PICKLE_OUT, 'wb') as f:
    pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)