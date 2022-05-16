import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def read_tensorboard(file_path):
    ea = event_accumulator.EventAccumulator(file_path)
    ea.Reload()
    train_loss, val_loss = ea.scalars.Items('train/loss'), ea.scalars.Items('test/loss')
    train_steps = []
    train_values = []
    for event in train_loss:
        train_steps.append(event.step)
        train_values.append(event.value)

    val_steps = []
    val_values = []
    for event in val_loss:
        val_steps.append(event.step)
        val_values.append(event.value)

    time = (train_loss[-1].wall_time - train_loss[0].wall_time) / 3600
    return train_steps, train_values, val_steps, val_values, time


def read_tensorboards(file_root):
    file_paths = []
    for d in os.listdir(file_root):
        d_path = os.path.join(file_root, d)
        if os.path.isfile(d_path):
            file_paths.append(d_path)

    total_train_steps = []
    total_train_values = []
    total_val_steps = []
    total_val_values = []
    total_time = 0
    for file_path in file_paths:
        train_steps, train_values, val_steps, val_values, time = read_tensorboard(file_path)
        for i in range(len(train_steps)):
            if total_train_steps and train_steps[i] <= total_train_steps[-1]:
                continue
            else:
                total_train_steps += train_steps[i:]
                total_train_values += train_values[i:]
                total_val_steps += val_steps[i:]
                total_val_values += val_values[i:]
                total_time += time
                break
    return [(total_train_steps, total_train_values), (total_val_steps, total_val_values)], total_time


def to_image(x, y, color, label):
    plt.plot(x, y, color=color, label=label)
    plt.legend()
    plt.title('PMT')
    plt.grid()
    plt.xlabel("step")  # ,loc='right')
    plt.ylabel("loss")


def main(file_root):
    data, total_time = read_tensorboards(file_root)
    print(total_time)
    plt.figure()
    labels = ['train', 'loss']
    colors = ['r', 'g']
    for i, color in enumerate(colors):
        to_image(data[i][0], data[i][1], color, labels[i])
    root = os.path.join(file_root, 'pic')
    os.makedirs(root, exist_ok=True)
    plt.savefig(f"{root}/result.png")


if __name__ == '__main__':
    main('/home/user/encoder4editing/experiment/PMT_lr1e-4/logs')
