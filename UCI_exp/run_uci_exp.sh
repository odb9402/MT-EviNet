nohup python -u uci_exp_norm.py -p boston -m $1 > log/boston_$1.log &
nohup python -u uci_exp_norm.py -p yacht -m $1 > log/yacht_$1.log &
nohup python -u uci_exp_norm.py -p energy -m $1 > log/energy_$1.log &
wait
nohup python -u uci_exp_norm.py -p concrete -m $1 > log/concrete_$1.log &
nohup python -u uci_exp_norm.py -p wine -m $1 > log/wine_$1.log &
wait
nohup python -u uci_exp_norm.py -p kin8nm -m $1 > log/kin8nm_$1.log &
nohup python -u uci_exp_norm.py -p power -m $1 > log/power_$1.log &
wait
nohup python -u uci_exp_norm.py -p protein -m $1 > log/protein_$1.log 
nohup python -u uci_exp_norm.py -p navel -m $1 > log/navel_$1.log 
wait