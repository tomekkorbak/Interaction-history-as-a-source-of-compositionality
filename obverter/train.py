import argparse
import os
import random

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from egg import core
import neptune
from neptunecontrib.api.utils import get_filepaths

from common import data
from common.visual_classifier import Vision
from common.callbacks import NeptuneMonitor, EarlyStopperAccuracy
from obverter.agent import Agent, AgentWrapper
from obverter.callbacks import CompositionalityMetricObverter


def get_params() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=5,
                        help='Dimensionality of the "concept" space (default: 10)')
    parser.add_argument('--n_attributes', type=int, default=2,
                        help='Number of attributes (default: 2')
    parser.add_argument('--seed', type=int, default=171,
                        help="Random seed")
    parser.add_argument('--neptune_project', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)

    # Agent architecture
    parser.add_argument('--sender_hidden', type=int, default=200,
                        help='Size of the hidden layer of Sender (default: 200)')
    parser.add_argument('--receiver_hidden', type=int, default=200,
                        help='Size of the hidden layer of Receiver (default: 200)')
    parser.add_argument('--sender_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Sender (default: 50)')
    parser.add_argument('--receiver_embedding', type=int, default=50,
                        help='Dimensionality of the embedding hidden layer for Receiver (default: 50)')
    parser.add_argument('--rnn_cell', type=str, default='rnn')

    args = core.init(parser)
    print(args)
    return args


def entangled_loss(targets, receiver_output_1, receiver_output_2):
    acc_1 = (receiver_output_1.argmax(dim=1) == targets[:, 0]).detach().float()
    acc_2 = (receiver_output_2.argmax(dim=1) == targets[:, 1]).detach().float()
    loss_1 = F.cross_entropy(receiver_output_1, targets[:, 0], reduction="none")
    loss_2 = F.cross_entropy(receiver_output_2, targets[:, 1], reduction="none")
    acc = (acc_1 * acc_2).mean(dim=0)
    loss = loss_1 + loss_2
    return loss, {f'accuracy': acc.item(),
                  f'first_accuracy': acc_1.mean(dim=0).item(),
                  f'second_accuracy': acc_2.mean(dim=0).item()}


class ObverterGame(nn.Module):

    def __init__(self, agents, max_len, vocab_size, loss):
        super(ObverterGame, self).__init__()
        self.agents = agents
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.loss = loss

    def forward(self, sender_input, target):
        sender, receiver = random.sample(self.agents, k=2)
        with torch.no_grad():
            message = sender.decode(sender_input)
        output_1, output_2 = receiver(message)
        loss, logs = self.loss(target, output_1, output_2)
        return loss.mean(), logs


if __name__ == "__main__":
    opts = get_params()
    opts.on_slurm = os.environ.get('SLURM_JOB_NAME', False)
    core.util._set_seed(opts.seed)
    full_dataset, train, test = data.prepare_datasets()
    train_loader = DataLoader(train, batch_size=opts.batch_size, drop_last=False, shuffle=True)
    test_loader = DataLoader(test, batch_size=opts.batch_size, drop_last=False, shuffle=False)

    agents = [AgentWrapper(
            agent=Agent(opts.receiver_hidden, opts.n_features),
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.rnn_cell,
            obverter_loss=entangled_loss,
            vision_module=Vision.from_pretrained('vision_model.pth')
    ) for _ in range(2)]
    game = ObverterGame(agents=agents, max_len=2, vocab_size=opts.vocab_size, loss=entangled_loss)
    optimizer = torch.optim.Adam([{'params': agent.parameters(), 'lr': 1e-5} for agent in agents])
    neptune.init(
        project_qualified_name=opts.neptune_project or 'anonymous/anonymous',
        backend=neptune.OfflineBackend() if not opts.neptune_project else None)
    with neptune.create_experiment(params=vars(opts), upload_source_files=get_filepaths()) as experiment:
        trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader,
                               validation_data=test_loader,
                               callbacks=[
                                   CompositionalityMetricObverter(full_dataset, agents[0], opts, opts.vocab_size, prefix='1_'),
                                   CompositionalityMetricObverter(full_dataset, agents[1], opts, opts.vocab_size, prefix='2_'),
                                   NeptuneMonitor(),
                                   core.ConsoleLogger(print_train_loss=not opts.on_slurm),
                                   EarlyStopperAccuracy(threshold=0.99, field_name='accuracy', delay=5)
                               ])
        trainer.train(n_epochs=500_000)
