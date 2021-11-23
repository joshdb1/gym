import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

nInputs = 4 # == env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[nInputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])

_bRender = True

def basicPolicy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

def playOneStep(env, obs, model, lossFn):
    with tf.GradientTape() as tape:
        leftProba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > leftProba)
        yTarget = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(lossFn(yTarget, leftProba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads

def playMultipleEpisodes(env, nEpisodes, nMaxSteps, model, lossFn, bRender):
    allRewards = []
    allGrads = []

    for episode in range(nEpisodes):
        currentRewards = []
        currentGrads = []
        obs = env.reset()
        for step in range(nMaxSteps):
            obs, reward, done, grads = playOneStep(env, obs, model, lossFn) #env.step(action)
            currentRewards.append(reward)
            currentGrads.append(grads)
            if (bRender):
                env.render()
            if done:
                break

            

        allRewards.append(currentRewards)
        allGrads.append(currentGrads)

    # stats = np.mean(allRewards), np.std(allRewards), np.min(allRewards), np.max(allRewards)    
    return allRewards, allGrads

def discountRewards(rewards, discountFactor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discountFactor
    return discounted

def discountAndNormalizeRewards(allRewards, discountFactor):
    allDiscountedRewards = [discountRewards(rewards, discountFactor)
                            for rewards in allRewards]
    flatRewards = np.concatenate(allDiscountedRewards)
    rewardMean = flatRewards.mean()
    rewardStd = flatRewards.std()
    return [(discountedRewards - rewardMean) / rewardStd
            for discountedRewards in allDiscountedRewards]

nIterations = 150
nEpisodesPerUpdate = 10
nMaxSteps = 200
discountFactor = 0.95

optimizer = keras.optimizers.Adam(lr=0.01)
lossFn = keras.losses.binary_crossentropy

env = gym.make("CartPole-v1")

for iteration in range(nIterations):
    allRewards, allGrads = playMultipleEpisodes(
        env, nEpisodesPerUpdate, nMaxSteps, model, lossFn, _bRender)
    allFinalRewards = discountAndNormalizeRewards(allRewards, discountFactor)
    allMeanGrads = []
    for varIdx in range(len(model.trainable_variables)):
        meanGrads = tf.reduce_mean(
            [finalReward * allGrads[episodeIdx][step][varIdx]
             for episodeIdx, finalRewards in enumerate(allFinalRewards)
                 for step, finalReward in enumerate(finalRewards)], axis=0
        )
        allMeanGrads.append(meanGrads)
    optimizer.apply_gradients(zip(allMeanGrads, model.trainable_variables))
    # print(allMeanGrads)

# totals = []
# for episode in range(500):
#     episodeRewards = 0
#     obs = env.reset()
#     for step in range(200):
#         action = basicPolicy(obs)
#         obs, reward, done, info = playOneStep(env, obs, model, ) #env.step(action)
#         if (_bRender):
#             env.render()
#         episodeRewards += reward
#         if done:
#             # print("CartPole finished after", step, "steps.")
#             break

#     totals.append(episodeRewards)

# stats = np.mean(totals), np.std(totals), np.min(totals), np.max(totals)
# print(stats)
