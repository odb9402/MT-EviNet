# Improving evidential deep learning via multi task learning

README of the AAAI2022 submission.

- Synthetic data experiment.
> python synthetic_exp.py

- UCI experiments.
> ./install_bayesopt.sh
> python uci_exp_norm.py -p yacht

- DTA tasks
> python train_evinet.py -o test --type davis -f 0
> python train_evinet_bindingdb.py -o test

- Grad conflict
> python check_conflict.py -f 0 --type davis
> python check_conflict.py -f 0 --type davis --abl