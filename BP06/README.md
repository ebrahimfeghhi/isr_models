# Immediate serial recall (ISR) code 
Code for training RNNs on immediate serial recall tasks 


## Structure of the code
To train an RNN, use the **run.py** file. **run.py** is currently integerated with wandb, which uploads figures and other training information to the cloud. There are two checkpoint styles: **accuracy**, which stops training after the accuracy across all list lengths exceeds a user-defined threshold AND the mean accuracy across list lenghts does not improve after a certain number of epochs, and **simulation_one**, which replicates the figures from Botvinick and Plaut 2006 for simulation one when the stopping criteria is set to human performance levels (~0.58). To run an RNN, modify the **sweep.yaml** file with the desired arguments. To start a sweep, enter **wandb sweep sweep.yaml** and follow the instructions to start the sweep. You can alternatively perform a single run by entering **python run.py** with the desired arguments. 

**run.py** relies on a number of other scripts to function. The architecture of RNNs is defined in **RNNcell.py**, trials are generated according using the Pytorch dataset class in **dataset.py**, **run_test_trials.py** tests an RNN on trials (test trials are stored in the **test_set** folder). All analyses are stored under the **model_analysis** folder. 



