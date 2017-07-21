DeepFlappyBird
====

# DeepQ

https://github.com/yenchenlin/DeepLearningFlappyBird

论文中说的用4个帧，但这里只是把同一个帧复制了4次。

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

q function update只是这一行

    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

experience replay

# Q

Q-learning version without cnn: https://github.com/chncyhn/flappybird-qlearning-bot

用了三个变量：1）小鸟到水管的x，2）小鸟到水管的y，3）小鸟的垂直方向的速度

pygame写的小鸟游戏：https://github.com/sourabhv/FlapPyBird

http://sarvagyavaish.github.io/FlappyBirdRL/

https://github.com/SarvagyaVaish/FlappyBirdRL

两个变量：小鸟到水管的x, y
