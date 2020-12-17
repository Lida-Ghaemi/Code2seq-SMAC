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
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario
# --------------------------------------------------------------
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
    config.BATCH_SIZE = cfg['BATCH_SIZE']
    config.NUM_EPOCHS = cfg['NUM_EPOCHS']
    config.MAX_TARGET_PARTS = cfg['MAX_TARGET_PARTS']   
    model = Model(config)
    if ii>0: #for the case where reuse is True inside GA
        model.train2()
        results, precision, recall, f1, rouge = model.evaluate()

    else:#for the case where reuse is False inside GA-first indiv
        model.train1()
        ii=2
        results, precision, recall, f1, rouge = model.evaluate()
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
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)
    
    
   # print(config.load_path)
    ##########################SMAC##############################
    # logger = logging.getLogger("SVMExample")
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    BATCH_SIZE=UniformIntegerHyperparameter('BATCH_SIZE', 128, 512, default_value=128) 
    print("dash bashuvaaaaaaaaaaaaaaaaaaaaaaa")   
    NUM_EPOCHS =UniformIntegerHyperparameter("NUM_EPOCHS", 7, 11, default_value=7)
    MAX_TARGET_PARTS=UniformIntegerHyperparameter("MAX_TARGET_PARTS", 6, 11, default_value=6)
    cs.add_hyperparameters([BATCH_SIZE,NUM_EPOCHS,MAX_TARGET_PARTS])
    # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
    #kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
    #cs.add_hyperparameter(kernel)
    # Scenario object
    ############################--------------------------------------------
#     config.SAVE_EVERY_EPOCHS = 1
#     config.PATIENCE = 10
#     config.TEST_BATCH_SIZE = 256
#     config.READER_NUM_PARALLEL_BATCHES = 1
#     config.SHUFFLE_BUFFER_SIZE = 10000
#     config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
#     config.MAX_CONTEXTS = 200
#     config.SUBTOKENS_VOCAB_MAX_SIZE = 190000
#     config.TARGET_VOCAB_MAX_SIZE = 27000
#     config.EMBEDDINGS_SIZE = 128
#     config.RNN_SIZE = 128 * 2  # Two LSTMs to embed paths, each of size 128
#     config.DECODER_SIZE = 320
#     config.NUM_DECODER_LAYERS = 1
#     config.MAX_PATH_LENGTH = 8 + 1
#     config.MAX_NAME_PARTS = 5
#     config.MAX_TARGET_PARTS = 6
#     config.EMBEDDINGS_DROPOUT_KEEP_PROB = 0.75
#     config.RNN_DROPOUT_KEEP_PROB = 0.5
#     config.BIRNN = True
#     config.RANDOM_CONTEXTS = True
#     config.BEAM_WIDTH = 0
#     config.USE_MOMENTUM = True
#     config.TRAIN_PATH = args.data_path
#     config.TEST_PATH = args.test_path if args.test_path is not None else ''
#     config.DATA_NUM_CONTEXTS = 0
#     config.SAVE_PATH = args.save_path_prefix
#     config.LOAD_PATH = args.load_path
#     config.RELEASE = args.release
    ###################
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 5,  # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true"
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = mysmac_from_cfg(cs.get_default_configuration())
    print("Default Value: %.2f" % (def_value))

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=mysmac_from_cfg)

    incumbent = smac.optimize()

    inc_value = mysmac_from_cfg(incumbent)

    print("Optimized Value: %.2f" % (inc_value))

    # We can also validate our results (though this makes a lot more sense with instances)
    smac.validate(config_mode='inc',  # We can choose which configurations to evaluate
                  # instance_mode='train+test',  # Defines what instances to validate
                  repetitions=3,  # Ignored, unless you set "deterministic" to "false" in line 95
                  n_jobs=1)  # How many cores to use in parallel for optimization
    

    

   ##########################SMAC------end---------------##############################
#     config.BATCH_SIZE=best[0]
#       #config.RNN_SIZE =indiv[1]*2
#     config.NUM_EPOCHS =best[1]
#       #config.NUM_DECODER_LAYERS=indiv[2]
#     config.MAX_TARGET_PARTS=best[2]
      #model = Model(config)

     #def print_hyperparams(self):
    print('Training batch size:\t\t\t', config.BATCH_SIZE)
    print('Epochs:\t\t', config.NUM_EPOCHS)
    print('Max target length:\t\t\t', config.MAX_TARGET_PARTS)
    print('Dataset path:\t\t\t\t', config.TRAIN_PATH)
    print('Training file path:\t\t\t', config.TRAIN_PATH + '.train.c2s')
    print('Validation path:\t\t\t', config.TEST_PATH)
    print('Taking max contexts from each example:\t', config.MAX_CONTEXTS)
    print('Random path sampling:\t\t\t', config.RANDOM_CONTEXTS)
    print('Embedding size:\t\t\t\t', config.EMBEDDINGS_SIZE)
    if config.BIRNN:
        print('Using BiLSTMs, each of size:\t\t', config.RNN_SIZE // 2)
    else:
        print('Uni-directional LSTM of size:\t\t', config.RNN_SIZE)
    print('Decoder size:\t\t\t\t', config.DECODER_SIZE)
    print('Decoder layers:\t\t\t\t', config.NUM_DECODER_LAYERS)
    print('Max path lengths:\t\t\t', config.MAX_PATH_LENGTH)
    print('Max subtokens in a token:\t\t', config.MAX_NAME_PARTS)
    print('Embeddings dropout keep_prob:\t\t', config.EMBEDDINGS_DROPOUT_KEEP_PROB)
    print('LSTM dropout keep_prob:\t\t\t', config.RNN_DROPOUT_KEEP_PROB)
    print('============================================') 
    #aa=evaluate_each_indiv(model,config)
    #print("heyyyyyyyyyyyyyyyyy I am starting main train\n")
    
    model = Model(config)
    print("\n************************************* this is the config to train ************************************\n ")
    print(config.BATCH_SIZE,config.NUM_EPOCHS ,config.MAX_TARGET_PARTS)
      #model = Model(config)
    print('Created model')
    if config.TRAIN_PATH:
        model.train()
    if config.TEST_PATH and not args.data_path:
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

