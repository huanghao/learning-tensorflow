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

If we apply the same preprocessing to game screens as in the DeepMind paper – take the four last screen images, resize them to 84×84 and convert to grayscale with 256 gray levels – we would have 25684x84x4 ≈ 1067970 possible game states.
