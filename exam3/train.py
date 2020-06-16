import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model, dataset


@click.command()
@click.help_option('-h', '--help')
@click.option('--epochs', type=click.INT, default=20, help='Number of epochs.')
@click.option('--window', type=click.INT, default=22, help='Number of window size.')
@click.option('--step', type=click.INT, default=1, help='Number of step size.')
def train_model(epochs=10, window=22, step=1):
    dataset_tr = dataset(task='train', window_size=window, step_size=step)
    dataset_ev = dataset(task='evaluate', window_size=window, step_size=step)
    model = Model(dataset_tr.features())
    print(f'model built! It will training on {len(dataset_tr)} seqs and evaluate on {len(dataset_ev)} seqs')
    print(model)
    print(f"Trainable Perameters: {sum(p.numel() for p in model.parameters())}\n")

    opt = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    d_loader_tr = DataLoader(
        dataset=dataset_tr,
        shuffle=True,
        num_workers=2,
        batch_size = 128
    )

    d_loader_ev = DataLoader(
        dataset=dataset_ev,
        shuffle=True,
        num_workers=2,
        batch_size = 1024
    )

    losses = {'train': [], 'eval': []}
    best_loss = 1e9
    no_improve = 0
    for epoch in range(epochs):
        with tqdm(d_loader_tr, f'Epoch-#{epoch + 1}') as training:
            model.train()
            losses_tr = []
            for s, batch in enumerate(training):
                opt.zero_grad()

                out = model(batch['x'])

                loss = criterion(out, batch['y'].view(-1))
                loss.backward()
                opt.step()

                training.set_description(f'Epoch-#{epoch + 1} Loss={loss.item():.4f}')
                losses_tr.append(loss.item())
            losses['train'].append(np.mean(losses_tr))

        with torch.no_grad():
            model.eval()
            losses_ev = []
            for s, batch in enumerate(tqdm(d_loader_ev, 'Evaluating')):
                out = model(batch['x'])
                loss = criterion(out, batch['y'].view(-1))
                losses_ev.append(loss.item())
            losses['eval'].append(np.mean(losses_ev))

        print(f"Epoch-#{epoch + 1} AVG training loss: {losses['train'][-1]:.4f} / evaluation loss: {losses['eval'][-1]:.4f}")
        if losses['eval'][-1] <= best_loss:
            print(f"Evaluation loss improved form {best_loss} to {losses['eval'][-1]}, save best model.\n")
            best_loss = losses['eval'][-1]
            torch.save(model.state_dict(), './model')
            no_improve = 0
        else:
            no_improve += 1
            print(f'Model no improve for {no_improve} times.')


        if no_improve == 10:
            print('Model no improve reach 10 times. Early Stop!')
            break


    print('Training finished!')


if __name__ == '__main__':
    train_model()
