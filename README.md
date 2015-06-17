# NeuralNetworkDeterministicPolicyGradient

###Introduction

Neural Network Deterministic policy gradient is a Julia implementation of a 2014 MDP solution technique (http://jmlr.org/proceedings/papers/v32/silver14.pdf) that uses neural networks for the actor and the critic, eliminating the need for feature engineering.

###Installation
To install, simply run
```julia
Pkg.clone("https://github.com/cho3/NeuralNetworkDeterministicPolicyGradient")
```
Note: if you are behind a proxy server, you’ll need to configure git differently. The following commands should do it:
```julia
run(`git config --global http.proxy $http_proxy`) # where $http_proxy is your proxy server
run(`git config --global url."https://".insteadOf git://`) # forces git to use https
```

###Usage
Once installed, a generic use case is as follows:
```
using NeuralNetworkDeterministicPolicyGradient

#define problem
gm = GenerativeModel(init,getNext,isEnd,reward)

#initialize solver components
actor,critic,param,solver,updater = easyInit(n,ub,lb,mem_size,cv,cw=0.,cth=0.,ActorLayers=[5.],CriticLayers=[5.],neuron_type="relu")

#train a policy
policy, qs = train(gm,trainRNG,actor,critic,param,solver,updater,time_horizon=500,num_episodes=10,eps=0.5,alpha=[alpha_th;alpha_w;alpha_v],gamma=0.99,natural=true,verbose=false,experience_replay=false)

#evaluate policy
R_avg, hists = runSim(gm,simRNG,actRNG,policy,time_horizon=500,recordHist = false,nSims=100,verbosde=true)
```

Below is a brief overview of the less clear parameters
#####`GenerativeModel`

`init(AbstractRNG)`->`State`

`getNext(AbstractRNG,State,Action)`->`State`

`isEnd(State)`->`Bool`

`reward(State,Action`->`Float`

#####`easyInit`

`n`: dimensionality of state space

`ub`,`lb`: upper and lower bound of each action dimension (Float arrays)

`mem_size`: number of experience tuples to maintain in memory

`cv`,`cw`,`cth`: L2 regularization term on the value critic, advantage critic, and actor respectively

`ActorLayers`,`CriticLayers`: denotes the number of hidden layers and the size of each hidden layer as a factor of ```n```

`neuron_type`: activation function in the hidden and input layers

It is recommended that the states and actions be represented as vectors of floats.

A more complete, ready-to-run example can be found in ```tests/DPGTest.ipynb```
More extensive documentation may be found in the ```/tex``` folder.

[![Build Status](https://travis-ci.org/cho3/NeuralNetworkDeterministicPolicyGradient.jl.svg?branch=master)](https://travis-ci.org/cho3/NeuralNetworkDeterministicPolicyGradient.jl)
