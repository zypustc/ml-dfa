# coding=utf-8
from pyscf import dft
from benchmark_datasets import tools
import pyscf
import dftd4.pyscf as disp
from time import process_time
from pathlib import Path
import numpy as np
import time

def reaction_calculate(reaction_system, file_name, param, size):
    bs = "def2tzvpd"
    vb = 3
    gl = 4
    mc = 200
    atom_num = 0
    
    t1 = time.process_time()

    systems = reaction_system.systems
    reactions = reaction_system.reactions

    energy = dict()

    tools.input_set(systems)

    start_time = process_time()
    print(start_time)

    for item in systems:

        print(item)
        mole_info = tools.getAtom(item)
        print(mole_info)

        mole = pyscf.M(
            atom=mole_info[0],
            spin=mole_info[1],
            charge=mole_info[2],
            basis=bs,
            verbose=vb

        )

        if mole_info[1] == 0:
            ml_dfa = dft.RKS(mole)
            dfa = dft.RKS(mole)

        else:
            ml_dfa = dft.UKS(mole)
            dfa = dft.UKS(mole)

        ml_dfa.xc = 'b3lyp'
        ml_dfa.grids.level = gl
        ml_dfa.max_cycle = mc
        ml_dfa.level_shift = 0.4
        ml_dfa.conv_tol = 1e-8
        grid = dft.gen_grid.Grids(mole)
        grid.level = 4
        grid.build()
        dfa.grids = grid
        dfa.max_cycle = mc

        # �������ģ��
        res = 0
        ml_dfa_res = 0
        b3lyp_res = 0
        b3lyp_d_res = 0

        # dft-D energy
        d4 = disp.DFTD4Dispersion(mole, 'b3lyp')
        delta_d4 = d4.kernel()[0]
        print("delta_d4:", delta_d4)

        ml_res = ml_dfa.kernel()
        ml_d_res = ml_res + delta_d4
        ml_res = 0
        ml_d_res = 0

        b3lyp_res = dfa.kernel()
        b3lyp_d_res = b3lyp_res + delta_d4

        result = [ml_res, ml_d_res, delta_d4]
        energy[item] = result
        print("-------results-----------", result)

    ml_maes = []
    ml_d_maes = []
    d_data = []
    mae_strs = []

    for reaction in reactions:
        atom_nums = len(reaction['systems'])
        ml_res = 0
        ml_d_res = 0
        d_res = 0

        for i in range(atom_nums):
            ml_res += energy[reaction['systems'][i]][0] * int(reaction['stoichiometry'][i])
            ml_d_res += energy[reaction['systems'][i]][1] * int(reaction['stoichiometry'][i])
            d_res += energy[reaction['systems'][i]][2] * int(reaction['stoichiometry'][i])
            print("-------------")
            print(d_res)

        ref = reaction['reference']
        
        ml_diff =  ml_res * 627.5095 - ref
        ml_d_diff = ml_d_res * 627.5095 - ref

        ml_mae = round(abs(ml_res * 627.5095 - ref), 2)
        ml_d_mae = round(abs(ml_d_res * 627.5095 - ref), 2)
        print("ml_d_mae:", ml_d_mae)
        ml_result = round((ml_res * 627.5095 - ref), 2)
        ml_d_result = round((ml_d_res * 627.5095 - ref), 2)

        mae_str = str(ml_diff)+ " " + str(ml_d_diff)+ " " +str(ml_mae) + "  " + str(ml_d_mae) + "  "+str(d_res * 627.5095)

        ml_maes.append(ml_mae)
        ml_d_maes.append(ml_d_mae)
        d_data.append(d_res)
        mae_strs.append(mae_str)

        file_path = Path(f"/public/home/ypzhang/src/benchmark/output/{param}/set_loss/{file_name}_d.txt")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as file:
            for item in mae_strs:
                file.write(item + '\n')

    ml_loss = round(np.mean(np.asarray(ml_maes)), 2)
    ml_d_loss = round(np.mean(np.asarray(ml_d_maes)), 2)
    

    file_path_benchmark = Path(f"/public/home/ypzhang/src/benchmark/output/{param}/benchmark.txt")
    file_path_benchmark.parent.mkdir(parents=True, exist_ok=True)
    t2 = time.process_time()
    final_time = round(int(t2 - t1), 2)

    with open(file_path_benchmark, 'a') as file:
        file.write(file_name + " " + str(ml_loss) + " " + str(ml_d_loss) + " " + str(final_time) + '\n')