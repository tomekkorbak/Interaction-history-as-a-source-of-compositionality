import collections

import neptune
import torch

from common.callbacks import CompositionalityMetric


class CompositionalityMetricObverter(CompositionalityMetric):

    def run_inference(self):
        with torch.no_grad():
            ran_inference_on = collections.defaultdict(int)
            for (input, target) in self.dataset:
                target = tuple(target.tolist())
                if ran_inference_on[target] < 5:
                    message = self.sender.decode(input.unsqueeze(dim=0))[0]
                    message = tuple(message.tolist())
                    neptune.send_text(self.prefix + 'messages', f'{target} -> {message}')
                    self.input_to_message[target].append(message)
                    ran_inference_on[target] += 1
