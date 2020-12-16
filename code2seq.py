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
    #tf.set_random_seed(args.seed)
    #tf.random.set_seed(args.seed)
    ############################################################
    #print(args.debug)

    #if args.debug:
     #   config = Config.get_debug_config(args)
   # else:
    #    config = Config.get_default_config(args)
    config = Config.get_default_config(args)    
    #print(config.BATCH_SIZE)
    ##########################SMAC##############################
    
    

# --------------------------------------------------------------



#sys.path.append(os.path.join(os.path.dirname(__file__)))
#from mlp_from_cfg_func import mlp_from_cfg  # noqa: E402

logger = logging.getLogger("log")
logging.basicConfig(level=logging.INFO)
def svm_from_cfg(cfg):
    """ Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation.
    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!
    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    #if "gamma" in cfg:
      #  cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
     #   cfg.pop("gamma_value", None)  # Remove "gamma_value"

#    clf = svm.SVC(**cfg, random_state=42)
    model = Model(config)
    model.train()
    results, precision, recall, f1, rouge = model.evaluate()
    return f1

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()
    config.BATCH_SIZE=UniformIntegerHyperparameter('config.BATCH_SIZE', 128, 512, default_value=128)
    config.NUM_EPOCHS =UniformIntegerHyperparameter('config.NUM_EPOCHS', 7, 11, default_value=7)
    config.MAX_TARGET_PARTS=UniformIntegerHyperparameter('config.MAX_TARGET_PARTS', 6, 10, default_value=6)

    # We can add multiple hyperparameters at once:
    #n_layer = UniformIntegerHyperparameter("n_layer", 1, 5, default_value=1)
    #n_neurons = UniformIntegerHyperparameter("n_neurons", 8, 1024, log=True, default_value=10)
    #activation = CategoricalHyperparameter("activation", ['logistic', 'tanh', 'relu'],
    #                                       default_value='tanh')
    #batch_size = UniformIntegerHyperparameter('batch_size', 64, 256, default_value=64)
    #learning_rate_init = UniformFloatHyperparameter('learning_rate_init', 0.0001, 1.0, default_value=0.001, log=True)
    cs.add_hyperparameters([ batch_size])

    # SMAC scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                         "wallclock-limit": 100,  # max duration to run the optimization (in seconds)
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         #"limit_resources": True,  # Uses pynisher to limit memory and runtime
                         # Alternatively, you can also disable this.
                         # Then you should handle runtime and memory yourself in the TA
                         "cutoff": 30,  # runtime limit for target algorithm
                         #"memory_limit": 3072,  # adapt this to reasonable value for your hardware
                         })

    # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_iters = 5
    # intensifier parameters
    intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_iters, 'eta': 3}
    # To optimize, we pass the function to the SMAC-object
    smac = HB4AC(scenario=scenario, rng=np.random.RandomState(42),
                 tae_runner=mlp_from_cfg,
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

