# Federated Motor Imagery Classification for Privacy-Preserving Brain-Computer Interfaces

This is the PyTorch implementation of the following paper:

Tianwang Jia, Lubin Meng, Siyang Li, Jiajing Liu and Dongrui Wu. Federated Motor Imagery Classification for Privacy-Preserving Brain-Computer Interfaces.

## Usage

### Setup

Please see the `requirements.txt` for environment configuration.

```bash
pip install -r requirements.txt
```

### Datasets

Please use the `getdata.py` to download MI data from `moabb`.

```bash
cd ./data
python -u get_data.py
```

### Train and Test

Federated Learning approaches

Please use the following commands to train a model with federated learning strategy and perform leave-one-subject-out testing.

```bash
cd ./run_scripts
# Here are the commands for running various models with different federated learning approaches on distinct datasets:
bash MI1_eegnet_fedavg.sh > MI1_eegnet_fedavg.log
bash MI1_eegnet_fedprox.sh > MI1_eegnet_fedprox.log
bash MI1_eegnet_scaffold.sh > MI1_eegnet_scaffold.log
bash MI1_eegnet_moon.sh > MI1_eegnet_moon.log
bash MI1_eegnet_fedfa.sh > MI1_eegnet_fedfa.log
bash MI1_eegnet_ga.sh > MI1_eegnet_ga.log
## Our approach FedBS: 
bash MI1_eegnet_fedbs.sh > MI1_eegnet_fedbs.log
bash MI1_deepconvnet_fedavg.sh > MI1_deepconvnet_fedavg.log
bash MI1_shallowconvnet_fedavg.sh > MI1_shallowconvnet_fedavg.log
bash MI2_eegnet_fedavg.sh > MI2_eegnet_fedavg.log
bash MI3_eegnet_fedavg.sh > MI3_eegnet_fedavg.log
```
