import numpy as np
import pickle
import os

loss = np.zeros((4, 5))
MSE_1 = np.zeros((4, 5))
MSE_2 = np.zeros((4, 5))
L1 = np.zeros((4, 5))

directory = './grid-search-results-early/'

paths = [
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.0beta-FalseDale-Nonepath-20210330-075814",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.1beta-FalseDale-20210331-000125",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.2beta-FalseDale-20210330-235301",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.3beta-FalseDale-20210331-000453",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.4beta-FalseDale-20210330-235358",
        "model-0.0beta-hierarchicalmode-1600units-45tsteps-0.0005lr-FalseDale-20framesize-2000epochs-0.25gradclip-1e-06L1-Nonepath-4warmup-20210329-192457",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.1beta-FalseDale-20210331-012717",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.2beta-FalseDale-20210331-003312",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.3beta-FalseDale-20210331-004704",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.4beta-FalseDale-20210331-012403",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.0beta-FalseDale-Nonepath-20210329-170258",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.1beta-FalseDale-20210330-173514",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.2beta-FalseDale-20210331-011540",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.3beta-FalseDale-20210402-040547",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.4beta-FalseDale-20210401-235750",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-1700epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.0beta-FalseDale-20210331-180833",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-300epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.1beta-FalseDale-20210407-153936",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-800epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.2beta-FalseDale-20210406-095746",
        "model-hierarchicalmode-20framesize-45tsteps-4warmup-300epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.3beta-FalseDale-20210407-082544",
        "model-1600units-FalseDale-0.4beta-4warmup-0.0005lr-3.162277660168379e-07L1-hierarchicalmode-20framesize-2000epochs-0.25gradclip-45tsteps-20210401-065206"
]
paths = [
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.0beta-FalseDale-Nonepath-20210330-145957",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.1beta-FalseDale-20210401-225322",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.2beta-FalseDale-20210401-035258",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.3beta-FalseDale-20210329-104947",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.4beta-FalseDale-20210330-143620",
    "model-0.0beta-hierarchicalmode-1600units-45tsteps-0.0005lr-FalseDale-20framesize-2000epochs-0.25gradclip-1e-06L1-Nonepath-4warmup-20210330-065920",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.1beta-FalseDale-20210401-154749",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.2beta-FalseDale-20210401-045259",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.3beta-FalseDale-20210401-144421",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.4beta-FalseDale-20210401-152633",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.0beta-FalseDale-Nonepath-20210330-012544",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.1beta-FalseDale-20210331-132239",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.2beta-FalseDale-20210401-053908",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.3beta-20210327-032254",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.4beta-FalseDale-20210402-103009",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1700epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.0beta-FalseDale-20210331-022004",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.1beta-20210327-095531",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.2beta-20210327-145725",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.3beta-FalseDale-20210402-081632",
    "model-1600units-FalseDale-0.4beta-4warmup-0.0005lr-3.162277660168379e-07L1-hierarchicalmode-20framesize-2000epochs-0.25gradclip-45tsteps-20210331-012222"
]

for i, path in enumerate(paths):
	with open(directory + path + '.pickle', 'rb') as p:
		loss_data = pickle.load(p)

		row = int(i/5)
		col = i % 5

		print(row, col)

		loss[row, col] = loss_data["loss"]
		MSE_1[row, col] = loss_data["MSE_1"]
		MSE_2[row, col] = loss_data["MSE_2"]
		L1[row, col] = loss_data["L1"]

		i += 1

print(loss)
print('\n')
print(MSE_1)
print('\n')
print(MSE_2)
print('\n')
print(L1)
print('\n')
print(MSE_1+MSE_2)
