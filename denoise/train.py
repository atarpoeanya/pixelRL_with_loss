from mini_batch_loader_c import *
from chainer import serializers
from MyFCN import *
from chainer import cuda, optimizers, Variable
import sys
import Loss
import torch
import math
import time
import chainerrl
from util import smallFunc
from state_c import State
import os
from pixelwise_a3c import *

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "../training_BSD68.txt"
TESTING_DATA_PATH           = "../testing.txt"
IMAGE_DIR_PATH              = "../"
SAVE_PATH            = "./model/denoise_myfcn_"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 5
SNAPSHOT_EPISODES  = 3000
TEST_EPISODES = 3000
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0.5

N_ACTIONS = 9
MOVE_RANGE = 3 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 70

W_SPA = 0
W_TV = 0
W_EXP = 0
W_COL_RATE = 0

GAMMA = 0
MULTI = 1
CLIP = 0.7
SIGMA = 0.4

GPU_ID = 0

def test(loader, agent, fout):
    sum_psnr     = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/255
        current_state.reset(raw_x,raw_n)
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
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        I = (I*255+0.5).astype(np.uint8)
        p = (p*255+0.5).astype(np.uint8)
        sum_psnr += cv2.PSNR(p, I)
 
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

    current_state = State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
 
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()
    
    #_/_/_/ training _/_/_/
 
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    # Loss function init
    L_spa = Loss.L_spa()
    L_TV = Loss.L_TV()
    L_exp = Loss.L_exp(16, 0.6)
    L_color_rate = Loss.L_color_rate()

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
            raw_n[i] = smallFunc.lower_contrast_batch(raw_n[i])
        # raw_n = 
        # b = 0
        # b += int(round(255*(1-MEAN)/2))
        # raw_n = cv2.addWeighted(raw_x.copy(), MEAN, raw_x.copy(), 0, b)

        # initialize the current state and reward
        current_state.reset(raw_n)
        reward = np.zeros(raw_n.shape, raw_n.dtype)
        sum_reward = 0
        
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()

            current_state.set(raw_x)
            action = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, gamma=GAMMA, multi=MULTI,clip=CLIP,SIGMA= SIGMA)

            raw_tensor = torch.from_numpy(raw_x).cuda()


            previous_image_tensor = torch.from_numpy(previous_image).cuda()
            current_state_tensor = torch.from_numpy(current_state.image).cuda()

            action_tensor = torch.from_numpy(action).cuda()

            # LOSS
            loss_spa_cur = W_SPA * torch.mean(L_spa(current_state_tensor, raw_tensor))
            Loss_TV_cur = W_TV * L_TV(action_tensor)
            loss_exp_cur = W_EXP * torch.mean(L_exp(current_state_tensor))
            # loss_col_rate_pre = W_COL_RATE * torch.mean(L_color_rate(previous_image_tensor, current_state_tensor))

            # reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
            # sum_reward += np.mean(reward)*np.power(GAMMA,t)
             # REWARD DECLARATION
            reward_current = loss_spa_cur + loss_exp_cur + Loss_TV_cur
            reward = - reward_current
            reward_de = reward.cpu().numpy()
            sum_reward += np.mean(reward_de) * np.power(GAMMA, t)


        agent.stop_episode_and_train(current_state.image, reward_de, True)
        print("train total reward {a}".format(a=sum_reward*255))
        fout.write("train total reward {a}\n".format(a=sum_reward*255))
        sys.stdout.flush()

        if episode % TEST_EPISODES == 0:
            #_/_/_/ testing _/_/_/
            test(mini_batch_loader, agent, fout)

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))
        
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
