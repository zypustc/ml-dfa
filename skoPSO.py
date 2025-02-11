import os.path
from datetime import datetime
from sko.PSO import PSO
from pso_utils import *
from pyscf import dft
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import validation_train.train_sets as train_system
import validation_train.test_sets as test_system
from gmtkn55 import tools
import pyscf
import dftd4.pyscf as disp
import sys
from validation_calculate import reaction_calculate
import time

# 定义神经网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, model_param, bias=True),
            nn.Sigmoid(),
            nn.Linear(model_param, model_param, bias=True),
            nn.Sigmoid(),
            nn.Linear(model_param, model_param, bias=True),
            nn.Sigmoid(),
            nn.Linear(model_param, 1, bias=True)
        )
        self.initialize_weights()  

    def forward(self, ml_in, is_training_data=False):
        inputs = torch.zeros((ml_in.shape[0], 3))
        inputs[:, 0] = torch.log(torch.pow((ml_in[:, 0] + ml_in[:, 1] + 1e-15), 1 / 3))
        inputs[:, 1] = torch.log(
            torch.div((ml_in[:, 0] - ml_in[:, 1]), (ml_in[:, 1] + ml_in[:, 0] + 1e-15)) + 1 + 1e-15)
        inputs[:, 2] = torch.log(torch.div(torch.pow((ml_in[:, 2] + ml_in[:, 4] + 2 * ml_in[:, 3] + 1e-15), 0.5),
                                           torch.pow((ml_in[:, 0] + ml_in[:, 1] + 1e-15), 4 / 3)))

        inputs = inputs.clone().detach().double()

        predict = self.model(inputs)
        # exc = predict * uni
        exc = predict

        return exc
        
    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                if layer.bias is not None:  # 检查偏置项是否存在
                    nn.init.constant_(layer.bias, 0)  # 偏置初始化为0
                nn.init.constant_(layer.weight, 0)  # 权重初始化为0

# TODO check 一次这段代码对不对
def eval_xc_ml(xc_code, rho, spin, relativity=0, deriv=1, verbose=None):
    # Definition of ML-DFA
    if spin == 0:
        rho0, dx, dy, dz = rho[:4]
        gamma1 = gamma2 = gamma12 = (dx ** 2 + dy ** 2 + dz ** 2) * 0.25 + 1e-10
        rho01 = rho02 = rho0 * 0.5
    else:
        rho1 = rho[0]
        rho2 = rho[1]
        rho01, dx1, dy1, dz1 = rho1[:4]
        rho02, dx2, dy2, dz2 = rho2[:4]
        gamma1 = dx1 ** 2 + dy1 ** 2 + dz1 ** 2 + 1e-10
        gamma2 = dx2 ** 2 + dy2 ** 2 + dz2 ** 2 + 1e-10
        gamma12 = dx1 * dx2 + dy1 * dy2 + dz1 * dz2 + 1e-10

    ml_in_ = np.concatenate((rho01.reshape((-1, 1)), rho02.reshape((-1, 1)), gamma1.reshape((-1, 1)),
                             gamma12.reshape((-1, 1)), gamma2.reshape((-1, 1))), axis=1)
    N = rho01.shape[0]

    ml_in = Variable(torch.Tensor(ml_in_), requires_grad=True)

    exc_ml_out = model(ml_in)

    ml_exc = exc_ml_out.data[:, 0].numpy()
    exc_ml = torch.dot(exc_ml_out[:, 0].float(), ml_in[:, 0] + ml_in[:, 1])
    exc_ml.backward()
    grad = ml_in.grad.data.numpy()

    if spin != 0:
        vrho_ml = np.hstack((grad[:, 0].reshape((-1, 1)), grad[:, 1].reshape((-1, 1))))
        vgamma_ml = np.hstack(
            (grad[:, 2].reshape((-1, 1)), grad[:, 4].reshape((-1, 1)), grad[:, 3].reshape((-1, 1))))

    else:
        vrho_ml = (grad[:, 0] + grad[:, 1]) * 0.5
        vgamma_ml = (grad[:, 2] + grad[:, 3] + grad[:, 4]) * 0.25

    # Mix with existing functionals
    b3lyp_xc = dft.libxc.eval_xc('B3LYP', rho, spin, relativity, deriv, verbose)
    b3lyp_exc = np.array(b3lyp_xc[0])
    b3lyp_vrho = np.array(b3lyp_xc[1][0])
    b3lyp_vgamma = np.array(b3lyp_xc[1][1])
    exc = b3lyp_exc + ml_exc.flatten()
    vrho = b3lyp_vrho + vrho_ml
    vgamma = b3lyp_vgamma + vgamma_ml

    vlapl = b3lyp_xc[1][2]
    vtau = b3lyp_xc[1][3]
    vxc = (vrho, vgamma, vlapl, vtau)
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative

    return exc, vxc, fxc, kxc

    # 适应度函数


# 自定义损失函数
def loss(params, reaction_system):
    indicator = 0

    update_model_parameters(model, params, 0.0005)

    # 在这里把参数传过来，就可以直接更新model了
    indicator = 0
    systems = reaction_system.systems
    tools.input_set(systems)
    energy = dict()

    # TODO system能量计算
    for item in systems:
        mole_info = tools.getAtom(item)
        print(item)
        
        mole = pyscf.M(
            atom=mole_info[0],
            charge=mole_info[2],
            spin=mole_info[1],
            basis=bs,
            verbose=vb)

        if mole_info[1] == 0:
            ml_dfa = dft.RKS(mole)
        else:
            ml_dfa = dft.UKS(mole)
        ml_dfa = ml_dfa.define_xc_(eval_xc_ml, 'GGA', hyb=0.2, rsh=[0.0, 0.0, 0.2])

        ml_dfa.grids.level = gl
        ml_dfa.max_cycle = mc

        d4 = disp.DFTD4Dispersion(mole, 'b3lyp')
        delta_d4 = d4.kernel()[0]

        # result
        ml_res = ml_dfa.kernel() + (delta_d4 / 627.5095)
        # ml_d_res = ml_res + delta_d4
        ml_d_res = 0

        if not ml_dfa.converged:
            indicator = 1

        result = [ml_res, ml_d_res]
        energy[item] = result

    reactions = reaction_system.reactions

    ml_maes = []

    # TODO 反应误差统计
    for reaction in reactions:
        print(reaction)
        atom_nums = len(reaction['systems'])
        ml_res = 0

        for i in range(atom_nums):
            ml_res += energy[reaction['systems'][i]][0] * int(reaction['stoichiometry'][i])

        ref = reaction['reference']

        ml_mae = abs(ml_res * 627.5095 - ref)

        ml_maes.append(ml_mae)

    train_loss = np.sum(ml_maes)

    return train_loss.item(), indicator  # 返回损失值


def pso_fitness_function(params):
    global train_loss
    if len(train_loss_values) == 0:
        params = torch.zeros(total_params, dtype=torch.double, requires_grad=True)
        train_loss, converge_flag = loss(params, train_system)
        print("train_loss_begin:", train_loss)
        train_loss_values.append(train_loss)
        print("Here is train_loss:", train_loss)
    else:
        train_loss_param, converge_flag = loss(params, train_system)
        print("train_loss:", train_loss_param)

        train_loss_values.append(train_loss_param)  # 保存训练损失
        
    
        if  train_loss_param < train_loss:
            
            param_str = str(train_loss_param) + "_" + str(time.time())
            output_parameters(model, "{}/train_param".format(loss_path),"{}.npz".format(param_str))
            # TODO 在这里加一个计算validation_loss 的方法
            output_path = './{}'.format(loss_path)
            reaction_calculate("{}/train_param/{}.npz".format(output_path,param_str), 40, output_path)
            
            

    if len(train_loss_values) % 10 == 0:
        # 将所有的 train_loss 值追加到文件中
        with open("./{}/train_loss.txt".format(loss_path), "a") as file:
            for loss_value in train_loss_values:
                file.write(f"{loss_value}\n")

        all_train_loss_values.extend(train_loss_values)
        train_loss_values.clear()
        

    # 返回训练损失
    return train_loss


# PSO优化
global model_param 
global loss_path
model_param = 40
total_params = (model_param * 3) + model_param + (model_param * model_param) + model_param + (model_param * model_param) + model_param + (1 * model_param) + 1
# total_params = (20 * 3) + 20 + (20 * 20) + 20 + (20 * 20) + 20 + (1 * 20) + 1
train_loss_values = []  # 用于存储训练损失
val_loss_values = []  # 用于存储验证损失
all_train_loss_values = []
validation_interval = 100  # 每 100 次迭代更新验证损失
lb = [-5] * total_params  # 下界
ub = [5] * total_params  # 上界
bs = "def2tzvpd"
vb = 3
gl = 5
mc = 100
atom_num = 0
scale_factor =  0.0005
loss_path = ""
train_loss = 0
train_loss_param = 0

train_set_no = sys.argv[1]
train_group = sys.argv[2]


if __name__ == "__main__":
    begin_time = datetime.now().date()

    save_path = "train_set" + "_" + str(train_set_no) + "_" +str(train_group)

    loss_path = "./train_data/{}".format(save_path)
    print(loss_path)

    if not os.path.exists(loss_path):
        os.mkdir(loss_path)
        os.mkdir("{}/train_param".format(loss_path))
        os.mkdir("{}/validation".format(loss_path))

    # 创建模型
    model = SimpleNN()

    pso = PSO(func=pso_fitness_function, dim=total_params, pop=20, max_iter=10000, ub=ub, lb=lb)
    pso.run()


    # 获取最佳参数
    best_params = pso.gbest_x

    # 创建模型并设置最佳参数
    update_model_parameters(model, best_params)

    # 输出神经网络的参数
    output_parameters(model, loss_path, 'model_parameters.npz')

    # 保存loss
    end_time = datetime.now().date()
    save_loss_to_excel(train_loss_values, val_loss_values,
                       "./loss/train_begin@{}_end@{}.xlsx".format(begin_time, end_time))

