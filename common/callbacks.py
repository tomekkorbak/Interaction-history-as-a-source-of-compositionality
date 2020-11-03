import collections
import functools

import pandas as pd
from neptunecontrib.monitoring.utils import send_figure
import neptune
import matplotlib.pyplot as plt
from egg.core import Callback, EarlyStopperAccuracy
import torch
from torch.utils.data import DataLoader
from tabulate import tabulate
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns

from common.metrics import compute_concept_symbol_matrix, compute_context_independence, compute_representation_similarity
from template_transfer.games import CompositionalGameGS

class NeptuneMonitor(Callback):

    def __init__(self, prefix=None):
        self.epoch_counter = 0
        self.prefix = prefix + '_' if prefix else ''

    def on_epoch_end(self, loss, rest):
        self.epoch_counter += 1
        neptune.send_metric(f'{self.prefix}train_loss', self.epoch_counter, loss)
        for metric, value in rest.items():
            neptune.send_metric(f'{self.prefix}train_{metric}', self.epoch_counter, value)

    def on_test_end(self, loss, rest):
        neptune.send_metric(f'{self.prefix}test_loss', self.epoch_counter, loss)
        for metric, value in rest.items():
            neptune.send_metric(f'{self.prefix}test_{metric}', self.epoch_counter, value)


class CompositionalityMetric(Callback):

    def __init__(self, dataset, sender, opts, vocab_size, prefix=''):
        self.dataset = dataset
        self.sender = sender
        self.epoch_counter = 0
        self.opts = opts
        self.vocab_size = vocab_size
        self.prefix = prefix

        self.epoch_counter = 0

    def on_epoch_end(self, *args):
        self.epoch_counter += 1
        if self.epoch_counter % 10 == 0:
            self.input_to_message = collections.defaultdict(list)
            self.message_to_output = collections.defaultdict(list)
            train_state = self.trainer.game.training  # persist so we restore it back
            self.trainer.game.train(mode=False)
            for _ in range(10):
                self.run_inference()
            self.concept_symbol_matrix, concepts = compute_concept_symbol_matrix(
                self.input_to_message,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes,
                vocab_size=self.vocab_size
            )
            self.trainer.game.train(mode=train_state)
            self.print_table_input_to_message()
            self.draw_concept_symbol_matrix()

            # Context independence metrics
            context_independence_scores, v_cs = compute_context_independence(
                self.concept_symbol_matrix,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes,
            )
            neptune.send_metric(self.prefix + 'context independence', self.epoch_counter, context_independence_scores.mean(axis=0))
            neptune.send_text(self.prefix + 'v_cs', str(v_cs.tolist()))
            neptune.send_text(self.prefix + 'context independence scores', str(context_independence_scores.tolist()))

            # RSA
            correlation_coeff, p_value = compute_representation_similarity(
                self.input_to_message,
                input_dimensions=[self.opts.n_features] * self.opts.n_attributes
            )
            neptune.send_metric(self.prefix + 'RSA', self.epoch_counter, correlation_coeff)
            neptune.send_metric(self.prefix + 'RSA_p_value', self.epoch_counter, p_value)

    def on_train_end(self):
        self.on_epoch_end(self)

    def run_inference(self):
        raise NotImplementedError()

    def print_table_input_to_message(self):
        table_data = [['x'] + list(range(self.opts.n_features))] + [[i] + [None] * self.opts.n_features for i in range(self.opts.n_features)]
        for (input1, input2), messages in self.input_to_message.items():
            table_data[input1 + 1][input2 + 1] = '  '.join((' '.join((str(s) for s in message)) for message in set(messages)))
        for a, b in zip(range(self.opts.n_features), range(self.opts.n_features)):
            if a == b:
                table_data[a+1][(b % self.opts.n_features) + 1] = '*' + table_data[a+1][(b % self.opts.n_features) +1]
        filename = f'{self.prefix}input_to_message_{self.epoch_counter}.txt'
        with open(file=filename, mode='w', encoding='utf-8') as file:
            file.write(tabulate(table_data, tablefmt='fancy_grid'))
        neptune.send_artifact(filename)
        with open(file='latex' + filename, mode='w', encoding='utf-8') as file:
            file.write(tabulate(table_data, tablefmt='latex'))
        neptune.send_artifact('latex' + filename)

    def draw_concept_symbol_matrix(self):
        figure, ax = plt.subplots(figsize=(20, 5))
        figure.suptitle(f'Concept-symbol matrix {self.epoch_counter}')
        g = sns.heatmap(self.concept_symbol_matrix, annot=True, fmt='.2f', ax=ax)
        g.set_title(f'Concept-symbol matrix {self.epoch_counter}')
        send_figure(figure, channel_name=self.prefix + 'concept_symbol_matrix')
        plt.close()


class CompositionalityMetricGS(CompositionalityMetric):

    def run_inference(self):
        with torch.no_grad():
            ran_inference_on = collections.defaultdict(int)
            for (input, target) in self.dataset:
                target = tuple(target.tolist())
                if ran_inference_on[target] < 5:
                    message = self.sender(input.unsqueeze(dim=0))[0]
                    message = tuple(message.argmax(dim=1).tolist())
                    neptune.send_text(self.prefix + 'messages', f'{target} -> {message}')
                    self.input_to_message[target].append(message)
                    ran_inference_on[target] += 1


class EarlyStopperAccuracy(EarlyStopperAccuracy):

    def __init__(self, threshold: float, field_name: str = 'acc', delay=5, train: bool = True) -> None:

        super(EarlyStopperAccuracy, self).__init__(threshold, field_name)
        self.delay = delay
        self.train = train

    def should_stop(self) -> bool:
        data = self.train_stats if self.train else self.validation_stats
        if len(data) < self.delay:
            return False
        assert data is not None, 'Validation/Train data must be provided for early stooping to work'
        return all(logs[self.field_name] > self.threshold for _, logs in data[-self.delay:])

    def on_train_end(self):
        if self.should_stop():
            print(f'Stopped early on epoch {self.epoch}')


class WeightDumper(Callback):

    def __init__(self, game, dataset, label):
        self.game = game
        self.dataset = dataset
        self.label = label

    def on_epoch_end(self, *args):
        self.run_inference()
        self.visualize_embeddings()

    def visualize_embeddings(self):
        embeddings = self.game.receiver.embedding.weight.detach().transpose(1, 0)
        pca = PCA(n_components=2)
        embeddings_projected = pca.fit_transform(embeddings)
        np.savetxt('embs.txt', embeddings_projected)
        neptune.send_artifact('embs.txt')
        ax = sns.scatterplot(x=embeddings_projected[:, 0], y=embeddings_projected[:, 1])
        for i in range(10):
            ax.annotate(str(i), embeddings_projected[i], size=20)
        sns.despine(left=True, bottom=True)
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        figure = ax.get_figure()
        send_figure(figure, channel_name='embeddings')
        figure.savefig('figx.png')
        plt.close(figure)

    def run_inference(self):
        self.activations = collections.defaultdict(list)

        def log_activation(module, input, output, targets):
            self.activations[targets].append(input[0].squeeze(dim=0))

        with torch.no_grad():
            ran_inference_on = collections.defaultdict(int)
            for input, target in self.dataset:
                if ran_inference_on[tuple(target.tolist())] < 1:
                    func = functools.partial(log_activation, targets=tuple(target.tolist()))
                    handle = self.game.receiver.agent.register_forward_hook(func)
                    self.game.eval()
                    self.game(input.unsqueeze(dim=0), target.unsqueeze(dim=0))
                    handle.remove()
                    ran_inference_on[tuple(target.tolist())] += 1
        self.visualize(PCA(n_components=2))

    def visualize(self, dimensionality_reduction_transform):
        *_, last_activations = zip(
            *[activations for (key, activations) in sorted(self.activations.items())]
        )
        activations_projected = dimensionality_reduction_transform.fit_transform(
            np.vstack(last_activations)
        )
        df = pd.DataFrame.from_dict({
            'color': (['blue']*5 + ['cyan']*5 + ['gray']*5 + ['green']*5 + ['magenta']*5),
            'shape': ['box', 'sphere', 'cylinder', 'torus', 'ellipsoid']*5,
            'x': activations_projected[:, 0],
            'y': activations_projected[:, 1]
        })
        sns.set(style="whitegrid")
        ax = sns.scatterplot(x="x", y="y", hue="color", style="shape", data=df, legend=False,
                             palette=dict(blue='blue', cyan='cyan', gray='gray', green='green', magenta='magenta'),
                             markers=('s', 'o', 'D', 'X', '^'), s=70)
        sns.despine(left=True, bottom=True)
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        figure = ax.get_figure()
        send_figure(figure, channel_name=self.label)
        figure.savefig('fig.png')
        plt.close(figure)


class GradientDumper(Callback):

    def __init__(self, game, dataset, label):
        self.game = game
        self.dataset = dataset
        self.label = label

    def on_epoch_end(self, *args):
        loader = DataLoader(self.dataset, batch_size=500, drop_last=False, shuffle=True)
        input, _ = next(iter(loader))
        g1, g2 = self.get_gradients(input)
        neptune.send_text('color_grad', str(g1))
        neptune.send_text('shape_grad', str(g2))
        ax = sns.heatmap(torch.cat([g1, g2], dim=0), xticklabels=['color', 'shape'],
                         yticklabels=['$m_1$', '$m_2$'])
        figure = ax.get_figure()
        send_figure(figure, channel_name=f'{self.label} grads')
        figure.savefig('fig.png')
        plt.close(figure)

    def get_gradients(self, sender_input):
        message = self.game.sender(sender_input)
        message = message.detach()
        message.requires_grad = True
        first_receiver_output, second_receiver_output = self.game.receiver(message)
        first_receiver_output.sum().backward(retain_graph=True)
        first_classifier_grad_wrt_message = message.grad.mean(dim=0)
        message.grad.zero_()
        second_receiver_output.sum().backward()
        second_classifier_grad_wrt_message = message.grad.mean(dim=0)
        return first_classifier_grad_wrt_message, second_classifier_grad_wrt_message