# Interaction history as a source of compositionality in emergent communication

We assume Python <= 3.6. To reproduce the results of template transfer run:
```bash
pip install -r requirements.txt
unzip data.zip
python -m template_transfer.train
```

To reproduce reported baselines, run the following commands:
* Random baseline: `python -m template_transfer.train --sender_lr 0 --receiver_lr 0 --no_transfer`
* Same architecture without template transfer: `python -m template_transfer.train --no_transfer`
* Obverter: `python -m obverter.train`

Use `--help` flag for available arguments. All arguments default to hyperparameters used in the paper. We use [Neptune.ml](https://neptune.ml/) for experiment management, which is turned off by default. Pass `--neptune_project <username/projectname>` and set environmental variable `NEPTUNE_API_TOKEN` to log metrics using Neptune.
