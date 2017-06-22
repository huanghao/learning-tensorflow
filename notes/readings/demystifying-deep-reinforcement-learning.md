Demystifying Deep Reinforcement Learning
====

https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

roadmap
1. credit assignment problem & exploration-exploitation dilemma
2. MDP
3. discounted future reward
4. Q-learning
5. replace Q-table with a deep nn
6. experience replay technique
7. simple solution to exploration-exploitation problem

has ~sparse~ and ~time-delayed~ labels - the rewards

The main idea in Q-learning is that we can iteratively approximate the Q-function using the Bellman equation.

http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf

The obvious choice is screen pixels – they implicitly contain all of the relevant information about the game situation, except for the speed and direction of the ball. Two consecutive screens would have these covered as well.

If we apply the same preprocessing to game screens as in the DeepMind paper – take the four last screen images, resize them to 84×84 and convert to grayscale with 256 gray levels – we would have $256^{84x84x4} ≈ 10^1067970$ possible game states.

Neural networks are exceptionally good at coming up with good features for highly structured data.

People familiar with object recognition networks may notice that there are no pooling layers. But if you really think about it, pooling layers buy you translation invariance – the network becomes insensitive to the location of an object in the image. That makes perfectly sense for a classification task like ImageNet, but for games the location of the ball is crucial in determining the potential reward and we wouldn’t want to discard this information!

There is a whole bag of tricks that you have to use to actually make it converge. And it takes a long time, almost a week on a single GPU.

ε-greedy exploration

There are many more tricks that DeepMind used to actually make it work – like target network, error clipping, reward clipping etc

And we are using this garbage (the maximum Q-value of the next state) as targets for the network, only occasionally folding in a tiny reward.

Watching them figure out a new game is like observing an animal in the wild – a rewarding experience by itself.
