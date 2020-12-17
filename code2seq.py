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
from smac.scenario.scenario import 
# --------------------------------------------------------------
import os
import sys

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
    
    model = Model(cfg)
    model.train()
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
    
    
    #print(config.BATCH_SIZE)
    ##########################SMAC##############################
    # logger = logging.getLogger("SVMExample")
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    config.BATCH_SIZE=UniformIntegerHyperparameter('BATCH_SIZE', 128, 512, default_value=128)
    cs.add_hyperparameters([C, shrinking])
    config.NUM_EPOCHS =UniformIntegerHyperparameter("NUM_EPOCHS", 7, 11, default_value=7)
      
    config.MAX_TARGET_PARTS=UniformIntegerHyperparameter("MAX_TARGET_PARTS", 6, 11, default_value=6)

    # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
    #kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
    #cs.add_hyperparameter(kernel)

    # There are some hyperparameters shared by all kernels
    C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
    shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="true")
    cs.add_hyperparameters([C, shrinking])

    # Others are kernel-specific, so we can add conditions to limit the searchspace
    degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)  # Only used by kernel poly
    coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
    cs.add_hyperparameters([degree, coef0])
    use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
    use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
    cs.add_conditions([use_degree, use_coef0])

    # This also works for parameters that are a mix of categorical and values from a range of numbers
    # For example, gamma can be either "auto" or a fixed float
    gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
    gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
    cs.add_hyperparameters([gamma, gamma_value])
    # We only activate gamma_value if gamma is set to "value"
    cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
    # And again we can restrict the use of gamma in general to the choice of the kernel
    cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))

    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 50,  # max. number of function evaluations; for this example set to a low number
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
                  repetitions=100,  # Ignored, unless you set "deterministic" to "false" in line 95
                  n_jobs=1)  # How many cores to use in parallel for optimization
    

    

   ##########################SMAC------end---------------##############################
    config.BATCH_SIZE=best[0]
      #config.RNN_SIZE =indiv[1]*2
    config.NUM_EPOCHS =best[1]
      #config.NUM_DECODER_LAYERS=indiv[2]
    config.MAX_TARGET_PARTS=best[2]
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

