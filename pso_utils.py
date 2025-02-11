from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

def output_parameters(model, loss_path, file_name):
    np.savez("{}/{}".format(loss_path, file_name),
             fc1_weight=model.model[0].weight.data.numpy(),
             fc1_bias=model.model[0].bias.data.numpy(),
             fc2_weight=model.model[2].weight.data.numpy(),
             fc2_bias=model.model[2].bias.data.numpy(),
             fc3_weight=model.model[4].weight.data.numpy(),
             fc3_bias=model.model[4].bias.data.numpy(),
             fc4_weight=model.model[6].weight.data.numpy(),
             fc4_bias=model.model[6].bias.data.numpy()
             )


def plot_loss(loss_dict, loss_path):
    # 提取 current_step, train_loss 和 val_loss
    steps = list(loss_dict.keys())  # 横坐标：所有步骤
    train_losses = [loss_dict[step]['train_loss'] for step in steps]  # 纵坐标：训练损失
    val_losses = [loss_dict[step]['val_loss'] for step in steps]  # 纵坐标：验证损失

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制训练损失
    plt.plot(steps, train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')

    # 绘制验证损失
    plt.plot(steps, val_losses, label='Validation Loss', color='red', linestyle='-', marker='x')

    # 添加标题和标签
    plt.title('Train and Validation Loss Over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss Value')

    # 添加图例
    plt.legend()

    # 保存图像，每次根据当前步骤保存不同的图像
    plt.savefig("{}/loss_plot_step_{}.png".format(loss_path, steps))


    # 清空当前图形
    plt.clf()

def load_param_from_npz(file_path, model, scale_factor):


    param_data = np.load(file_path)

    model.model[0].weight.data = torch.tensor(param_data["fc1_weight"])
    model.model[0].bias.data = torch.tensor(param_data["fc1_bias"])
    model.model[2].weight.data = torch.tensor(param_data["fc2_weight"])
    model.model[2].bias.data = torch.tensor(param_data["fc2_bias"])
    model.model[4].weight.data = torch.tensor(param_data["fc3_weight"])
    model.model[4].bias.data = torch.tensor(param_data["fc3_bias"])


    model.model[6].weight.data = torch.tensor(param_data["fc4_weight"]) * scale_factor
    model.model[6].bias.data = torch.tensor(param_data["fc4_bias"]) * scale_factor


def update_model_parameters(model, best_params, scale_factor):
    with torch.no_grad():
        start = 0
        for i, layer in enumerate(model.model):
            if isinstance(layer, nn.Linear):
                end = start + layer.weight.numel()
                layer.weight = nn.Parameter(torch.tensor(best_params[start:end]).reshape(layer.weight.shape))

                # 检查当前层是否有偏置项
                if layer.bias is not None:  # 只有在存在偏置时才进行赋值
                    end = start + layer.bias.numel()
                    layer.bias = nn.Parameter(torch.tensor(best_params[start:end]))
                    start = end

                if i == 6:  # 第三层的索引为2
                    layer.weight.data *= scale_factor  # 应用缩放
                    layer.bias.data *= scale_factor


def save_loss_to_excel(train_loss_values, val_loss_values, loss_path):
    df = pd.DataFrame({
        'Iteration': np.arange(len(train_loss_values)),
        'Train Loss': train_loss_values,
        'Validation Loss': [val_loss if i % 100 == 0 else None for i, val_loss in enumerate(val_loss_values)]
    })
    df.to_excel(loss_path, index=False)

def print_model_parameters(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")