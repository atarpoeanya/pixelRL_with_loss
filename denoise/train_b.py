from mini_batch_loader_c import *
from chainer import serializers
from customFCN import *
from chainer import cuda, optimizers, Variable
import sys
import random
import torch
import math
import time
import chainerrl
import smallFunc
from state_b import State
import os
from pixelwise_a3c import *
import csv

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "/content/training_list/train_list.txt"
TESTING_DATA_PATH           = "/content/testing_list/testing_list.txt"
IMAGE_DIR_PATH              = "../"
SAVE_PATH            = "/content/model/denoise_myfcn_"

csv_path = "/content/training_res/"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES       = 6
EPISODE_LEN = 10
SNAPSHOT_EPISODES  = 500
TEST_EPISODES = 5
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0.5
N_ACTIONS = 13
MOVE_RANGE = 3 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 70

pretrained = False
weight = '/content/model_loss/denoise_myfcn_240/model.npz'

parameters = {
    'learning_rate': LEARNING_RATE,

    'DISCOUNT_FACTOR': GAMMA,

}


GPU_ID = 0

def test(loader, agent, fout):
    sum_psnr     = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE))
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        raw_n = raw_x.copy()

        for i in range(0,1):
            raw_n[i] = smallFunc.HE(raw_n[i], 0.9, random.uniform(0.6, 0.8))
        current_state.reset(raw_n)
        
        reward = np.zeros(raw_x.shape, raw_x.dtype)*255
        
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action = agent.act(current_state.image)
            current_state.step(action)
            # UNUSED
            reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent.stop_episode()
            
        I = np.maximum(0,raw_x)
        I = np.minimum(1,I)
        N = np.maximum(0,raw_n)
        N = np.minimum(1,N)
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        I = (I[0]*255+0.5).astype(np.uint8)
        N = (N[0]*255+0.5).astype(np.uint8)
        p = (p[0]*255+0.5).astype(np.uint8)
        p = np.transpose(p,(1,2,0))
        I = np.transpose(I,(1,2,0))
        N = np.transpose(N,(1,2,0))
        
        cv2.imwrite('/content/res/'+str(i)+'_input.png', N)
        cv2.imwrite('/content/res/'+str(i)+'_truth.png', I)
        cv2.imwrite('/content/res/'+str(i)+'_output.png', p)
        sum_psnr += cv2.PSNR(p, N)
 
    print("test total reward {a}, PSNR {b}".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    fout.write("test total reward {a}, PSNR {b}\n".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    sys.stdout.flush()
 
 
def main(fout):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    # chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE))

 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
 
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C(model, optimizer, EPISODE_LEN, GAMMA)
    if pretrained:
      chainer.serializers.load_npz(weight, agent.model)
    agent.model.to_gpu()
    
    #_/_/_/ training _/_/_/
 
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    r = indices[i:i+TRAIN_BATCH_SIZE]
    result_array = []
    for episode in range(1, N_EPISODES+1):
        # display current episode
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        # load images
        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_x = mini_batch_loader.load_training_data(r)
        raw_n = raw_x.copy()
        # generate noise
        for i in range(0,64):
            raw_n[i] = smallFunc.HE(raw_n[i], 0.9, random.uniform(0.6, 0.8))
        # initialize the current state and reward
        current_state.reset(raw_n)
        reward = np.zeros(raw_n.shape, raw_n.dtype)
        sum_reward = 0
        
        for t in range(0, EPISODE_LEN):
            # Get St (current state of image)
            previous_image = current_state.image.copy()
            # Predict and generate action map based on St
            action = agent.act_and_train(current_state.image, reward)
            # Run action based on generated map
            current_state.step(action)

            reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
            sum_reward += np.mean(reward)*np.power(GAMMA,t)
            reward_de = reward

        agent.stop_episode_and_train(current_state.image, reward_de, True)
        print("train total reward {a}".format(a=sum_reward*255))
        fout.write("train total reward {a}\n".format(a=sum_reward*255))
        result_array.append([episode , sum_reward])
        sys.stdout.flush()

        if episode % TEST_EPISODES == 0:
            #_/_/_/ testing _/_/_/
            test(mini_batch_loader, agent, fout)

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))
            # Open the file in write mode
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"training_{timestamp}.csv"

             # Ensure the folder exists, create if it doesn't
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)
            
            # Create the full path for the file
            filepath = os.path.join(csv_path, filename)

            with open(filepath, 'w', newline='') as csvfile:
                # Create a CSV writer object
                csvwriter = csv.writer(csvfile)
                 # Write the parameters section
                csvwriter.writerow(['learning_rate','DISCOUNT_FACTOR'])
                csvwriter.writerow([
                    parameters['learning_rate'],
                    parameters['DISCOUNT_FACTOR'],                    
                ])
                
                # Add a blank row as a separator (optional)
                csvwriter.writerow([])
                
                # Optionally, write a header
                csvwriter.writerow(['Time', 'Reward Value'])
                
                # Write the data
                csvwriter.writerows(result_array)
        
        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:        
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)
 
     
 
if __name__ == '__main__':
    try:
        fout = open('log.txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start)/60))
        fout.write("{s}[h]\n".format(s=(end - start)/60/60))
        fout.close()
    except Exception as error:
        print(error.message)
