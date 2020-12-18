from argparse import ArgumentParser
import numpy as np
#%tensorflow_version 1.x
import tensorflow as tf
#import geneticalgorithm as ga
#from geneticalgorithm import geneticalgorithm as ga
from config import Config
from interactive_predict import InteractivePredictor
from model import Model
##-----------------------------------------------------from SMAC ------------------------------
import logging

import numpy as np
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from sklearn import datasets
from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario
# --------------------------------------------------------------

import warnings

#import ConfigSpace as CS
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score, StratifiedKFold



from smac.facade.smac_bohb_facade import BOHB4HPO


#-----------------------------------------
import os
import sys

ii=0
def mysmac_from_cfg(cfg):
    
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    #cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    #if "gamma" in cfg:
      #  cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
     #   cfg.pop("gamma_value", None)  # Remove "gamma_value"

#    clf = svm.SVC(**cfg, random_state=42)
    #print('############## ')
    #print(config.BATCH_SIZE)
    config1.BATCH_SIZE = cfg['BATCH_SIZE']
    #print('###########   ')
    #print(config.BATCH_SIZE)
    config1.NUM_EPOCHS = cfg['NUM_EPOCHS']
    #print('###########   ')
   # print(type(config.NUM_EPOCHS))
    config1.MAX_TARGET_PARTS = cfg['MAX_TARGET_PARTS']  
    #print('###########   ')
    #print(config.MAX_TARGET_PARTS)
    model = Model(config1)
    
    global ii
    print("iiiiiiiiiiiiiiii     ")
    print(ii)
    if ii>0: #for the case where reuse is True inside GA
        model.train2()
        results, precision, recall, f1, rouge = model.evaluate()

    else:#for the case where reuse is False inside GA-first indiv
        model.train1()
        ii=2
        print("otheriiiiiiiiiiiiiiii     ")
        print(ii)
        results, precision, recall, f1, rouge = model.evaluate()
    ii=2
    return f1

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a smaller model '
                             'size.')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    if args.debug:
        config1 = Config.get_debug_config(args)
    else:
        config1 = Config.get_default_config(args)
    
    #logger = logging.getLogger("MLP-example")
   
   # print(config.load_path)
    ##########################SMAC##############################
    # logger = logging.getLogger("SVMExample")
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    BATCH_SIZE=UniformIntegerHyperparameter('BATCH_SIZE', 128, 512, default_value=128) 
    #print("dash bashuvaaaaaaaaaaaaaaaaaaaaaaa")   
    NUM_EPOCHS =UniformIntegerHyperparameter("NUM_EPOCHS", 7, 11, default_value=7)
    MAX_TARGET_PARTS=UniformIntegerHyperparameter("MAX_TARGET_PARTS", 6, 11, default_value=6)
    cs.add_hyperparameters([BATCH_SIZE,NUM_EPOCHS,MAX_TARGET_PARTS])
    # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
    #kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
    #cs.add_hyperparameter(kernel)
#     # Scenario object
#     scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
#                          "runcount-limit": 5,  # max. number of function evaluations; for this example set to a low number
#                          "cs": cs,  # configuration space
#                          "deterministic": "true"
#                          })

#     # Example call of the function
#     # It returns: Status, Cost, Runtime, Additional Infos
#     def_value = mysmac_from_cfg(cs.get_default_configuration())
#     print("Default Value: %.2f" % (def_value))

#     # Optimize, using a SMAC-object
#     print("Optimizing! Depending on your machine, this might take a few minutes.")
#     smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
#                     tae_runner=mysmac_from_cfg)

#     incumbent = smac.optimize()

#     inc_value = mysmac_from_cfg(incumbent)

#     print("Optimized Value: %.2f" % (inc_value))

#     # We can also validate our results (though this makes a lot more sense with instances)
#     smac.validate(config_mode='inc',  # We can choose which configurations to evaluate
#                   # instance_mode='train+test',  # Defines what instances to validate
#                   repetitions=3,  # Ignored, unless you set "deterministic" to "false" in line 95
#                   n_jobs=1)  # How many cores to use in parallel for optimization
   ##########################SMAC------end---------------##############################
    # SMAC scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                         "wallclock-limit": 10,  #100 max duration to run the optimization (in seconds)
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         "limit_resources": True,  # Uses pynisher to limit memory and runtime
                         # Alternatively, you can also disable this.
                         # Then you should handle runtime and memory yourself in the TA
                         "cutoff": 10,  #30 runtime limit for target algorithm
                         "memory_limit": 307,  # 3072adapt this to reasonable value for your hardware
                         })

    # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_iters = 50
    # intensifier parameters
    intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_iters, 'eta': 3}
    # To optimize, we pass the function to the SMAC-object
    smac = BOHB4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=mysmac_from_cfg,
                    intensifier_kwargs=intensifier_kwargs)  # all arguments related to intensifier can be passed like this

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
                                          instance='1', budget=max_iters, seed=0)[1]
    print("Value for default configuration: %.4f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                          budget=max_iters, seed=0)[1]
    print("Optimized Value: %.4f" % inc_value)
##################-----smac mlp-----###################
#     config.BATCH_SIZE=best[0]
#       #config.RNN_SIZE =indiv[1]*2
#     config.NUM_EPOCHS =best[1]
#       #config.NUM_DECODER_LAYERS=indiv[2]
#     config.MAX_TARGET_PARTS=best[2]
      #model = Model(config)

     #def print_hyperparams(self):
    print('Training batch size:\t\t\t', config1.BATCH_SIZE)
    print('Epochs:\t\t\t\t', config1.NUM_EPOCHS)
    print('Max target length:\t\t\t', config1.MAX_TARGET_PARTS)
    print('Dataset path:\t\t\t\t', config1.TRAIN_PATH)
    print('Training file path:\t\t\t', config1.TRAIN_PATH + '.train.c2s')
    print('Validation path:\t\t\t', config1.TEST_PATH)
    print('Taking max contexts from each example:\t', config1.MAX_CONTEXTS)
    print('Random path sampling:\t\t\t', config1.RANDOM_CONTEXTS)
    print('Embedding size:\t\t\t\t', config1.EMBEDDINGS_SIZE)
    if config1.BIRNN:
        print('Using BiLSTMs, each of size:\t\t', config1.RNN_SIZE // 2)
    else:
        print('Uni-directional LSTM of size:\t\t', config1.RNN_SIZE)
    print('Decoder size:\t\t\t\t', config1.DECODER_SIZE)
    print('Decoder layers:\t\t\t\t', config1.NUM_DECODER_LAYERS)
    print('Max path lengths:\t\t\t', config1.MAX_PATH_LENGTH)
    print('Max subtokens in a token:\t\t', config1.MAX_NAME_PARTS)
    print('Embeddings dropout keep_prob:\t\t', config1.EMBEDDINGS_DROPOUT_KEEP_PROB)
    print('LSTM dropout keep_prob:\t\t\t', config1.RNN_DROPOUT_KEEP_PROB)
    print('============================================') 
    #aa=evaluate_each_indiv(model,config)
    #print("heyyyyyyyyyyyyyyyyy I am starting main train\n")
    
    model = Model(config1)
    print("\n************************************* this is the config to train ************************************\n ")
    print(config1.BATCH_SIZE,config1.NUM_EPOCHS ,config1.MAX_TARGET_PARTS)
      #model = Model(config)
    print('Created model')
    if config1.TRAIN_PATH:
        model.train()
    if config1.TEST_PATH and not args.data_path:
        results, precision, recall, f1, rouge = model.evaluate()
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Rouge: ', rouge)
    if args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    if args.release and args.load_path:
        model.evaluate(release=True)
    model.close_session()

