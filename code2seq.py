from argparse import ArgumentParser
import numpy as np
#%tensorflow_version 1.x
import tensorflow as tf
#import geneticalgorithm as ga
#from geneticalgorithm import geneticalgorithm as ga
from config import Config
from interactive_predict import InteractivePredictor
from model import Model

#################################################
def evaluate_each_indiv(config,i):
    
    #print("i am in evaluate_ga$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #if config.TRAIN_PATH:
    model = Model(config)
    if i>0: #for the case where reuse is True inside GA
        model.train2()
        #if args.data_path:
        results, precision, recall, f1, rouge = model.evaluate()
        #print("i am out of evaluate$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Rouge: ', rouge)
    else:#for the case where reuse is False inside GA-first indiv
        model.train1()
        #if args.data_path:
        results, precision, recall, f1, rouge = model.evaluate()
        #print("i am out of evaluate$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Rouge: ', rouge)
        
    #if args.predict:
    #    predictor = InteractivePredictor(config, model)
      #  predictor.predict()
   # if args.release and args.load_path:
     #   model.evaluate(release=True)
   # print("i have finished f111111111yyyyyyyyy")
    model.close_session()
    return f1
####################################################
def mymutate(indiv,ind):
    #va={64,128,256, 512}
    myNUM_EPOCHS=[7,8,9,10,11]
    myTraining_batch_size=[128, 256, 512]
    #myLSTMs_size=[16, 32,64,128,256]
    #myNumber_of_Decoder_layers: any value
    #myMax_target_length={1-10}
    #for i in range(len(indiv)):
    
    if ind==0:
      d1=np.random.randint(0,high=3,dtype=int)
      indiv[0]=myTraining_batch_size[d1]
    elif ind==1:
      b1=np.random.randint(0,high=5,dtype=int)
      #indiv[1]=myLSTMs_size[b1]
      indiv[1]=myNUM_EPOCHS[b1]
    #elif ind==2:
     # indiv[2]=np.random.randint(1,high=4,dtype=int)
    else: 
      indiv[2]=np.random.randint(6,high=11,dtype=int)
      config.BATCH_SIZE=indiv[0]
      #config.RNN_SIZE =indiv[1]*2
      config.NUM_EPOCHS =indiv[1]
      #config.NUM_DECODER_LAYERS=indiv[2]
      config.MAX_TARGET_PARTS=indiv[2]
      #model = Model(config)
      indiv[3]=evaluate_each_indiv(config,1)
    #print(indiv)
    return indiv
###############################################################
def initialize_pop(popsize,n_var,config):
    #va={64,128,256, 512}
    myTraining_batch_size=[64, 128, 256, 512]
    #myLSTMs_size=[16, 32,64,128,256]
    myNUM_EPOCHS=[7,8,9,10,11]
    #myNumber_of_Decoder_layers: any value
    #myMax_target_length={1-10}
    pop=[]
    for i in range(popsize):
      
      indiv=[0]*n_var
      d1=np.random.randint(0,high=3,dtype=int)
      b1=np.random.randint(0,high=5,dtype=int)
      indiv[0]=myTraining_batch_size[d1]
      #indiv[1]=myLSTMs_size[b1]
      indiv[1]=myNUM_EPOCHS[b1]
      indiv[2]=np.random.randint(6,high=11,dtype=int)#myMax_target_length
      #indiv[3]=np.random.randint(1,high=10,dtype=int)
      config.BATCH_SIZE=indiv[0]
      #config.RNN_SIZE =indiv[1]*2
      config.NUM_EPOCHS =indiv[1]
      #config.NUM_DECODER_LAYERS=indiv[2]
      config.MAX_TARGET_PARTS=indiv[2]
      #model = Model(config)
      indiv[3]=evaluate_each_indiv(config,i)
      
      pop+=[indiv]
      #model.close_session()
    print("initialization finished")
    return pop
##################################################
def mycross(pop,cross_p,popsize,n_var):
  for i in range(np.math.ceil(popsize/2)):
    r=np.random.randint(low=0,high=popsize, size=2,dtype=int)#crossover selection for two indivi
    cpoint=np.random.randint(low=1,high=n_var, size=1,dtype=int)# croxover point
    temp1=pop[r[0]]
    temp2=pop[r[1]]
    temp1[cpoint[0]:]=pop[r[1]][cpoint[0]:]
    temp2[cpoint[0]:]=pop[r[0]][cpoint[0]:]
    config.BATCH_SIZE=temp1[0]
      #config.RNN_SIZE =indiv[1]*2
    config.NUM_EPOCHS =temp1[1]
      #config.NUM_DECODER_LAYERS=indiv[2]
    config.MAX_TARGET_PARTS=temp1[2]
      #model = Model(config)
   
    temp1[n_var]=evaluate_each_indiv(config,1)
    config.BATCH_SIZE=temp2[0]
      #config.RNN_SIZE =indiv[1]*2
    config.NUM_EPOCHS =temp2[1]
      #config.NUM_DECODER_LAYERS=indiv[2]
    config.MAX_TARGET_PARTS=temp2[2]
      #model = Model(config)
    temp2[n_var]=evaluate_each_indiv(config,1)
    
    if temp1[n_var]>pop[r[0]][n_var]:
      pop[r[0]]=temp1
    if temp2[n_var]>pop[r[1]][n_var]:
      pop[r[1]]=temp2
  return pop    

#################################################
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
    #tf.random.set_seed(args.seed)
    ############################################################
    #print(args.debug)

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)
    #print(config.BATCH_SIZE)
    ###GA
    #varbound=np.array([[0,10]]*3)
    #modelga=ga(function=f,dimension=3,variable_type='real',variable_boundaries=varbound)
    #modelga.run()
    #############

    pop=[]
    n_var=3
    popsize=7#5
    pop=initialize_pop(popsize,n_var+1,config)
    print(pop)
    n_iters=10#7
    p_mutate=0.3
    cross_p=0.6
    for i in range(n_iters):
        pop=mycross(pop,cross_p,popsize,n_var)# crossover
        for j in range(popsize):#loop for muration
            r=np.random.random(1)
            if r<p_mutate:
                temp=mymutate(pop[j], np.random.randint(4))
                if temp[n_var]>pop[j][n_var]:
                    pop[j]=temp
    pop = sorted(pop, key = lambda x:x[n_var-1]) 
    print(pop)
    best=pop[-1]

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

