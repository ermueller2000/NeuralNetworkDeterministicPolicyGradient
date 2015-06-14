#DPGTest.jl
#TODO: make my own MDP.jl that has generative model, and simulate
#TODO: Arrays are objects, make sure you're not passing the pointer when you should be making a copy...

include("NN.jl")
include("ActorCritic.jl")
include("DPG.jl")
#include("src\\LinDPG.jl")
include("NNDPG.jl")
include("MTNCAR.jl")

using MTNCAR, NNDPG

function polynomial_basis(s::Array{Float64,1})
  #rescale to [-1,1]
  x = (s[1]+0.3)./0.9
  v = s[2]./0.07
  return [1;x;v;x*v;x*x;v*v;x*x*x;v*v*v;x*v*v;x*x*v]
end

function fourier_basis(s::Array{Float64,1},d::Int,coupled::Bool=false)
  #rescale to [0,1]
  x = (s[1]+1.2)./1.8
  v = (s[2]+0.07)./0.14

  #no variable coupling
  if coupled
    phi = ones(d*2+1)
    for k = 1:d
      phi[2+(k-1)*2] = cos(pi*k*x)
      phi[3+(k-1)*2] = cos(pi*k*v)
    end

  else
    phi = ones(2^(d+1)+1)
    offset = 1
    for k = 0:d
      for l = 0:d
        phi[offset] = cos(pi*(k*x+l*v))
        offset += 1
      end
    end
  end
  return phi
end


#feature_function = (x)->[1;x[1];x[2];x[1]*x[2];x[1]*x[1];x[2]*x[2];x[1]*x[1]*x[1];x[1]*x[1]*x[2];x[1]*x[2]*x[2];x[2]*x[2]*x[2]] #polynomial basis
#feature_function = polynomial_basis
feature_function = (s)->fourier_basis(s,3,true)
s0 = init()
phi = zeros(length(feature_function(s0)))

#s = LinDPG.Solver()
#u = LinDPG.DPGUpdater()

#actor = LinDPGActor(feature_function,deepcopy(phi),zeros(length(phi)),[-1.],[1.])
#critic = LinDPGCritic(feature_function, deepcopy(phi),deepcopy(phi),zeros(length(phi)),zeros(length(phi)))
#p = LinDPGParam(100)

init_ = (rng)->init()
getNext_ = (rng,s,a)->nextState(s,a)
isEnd_ = isEnd
reward_ = reward

gm = GenerativeModel(init_,getNext_,isEnd_,reward_)
#=

actornn = buildNN([2;10;10;1],0.001,-1.,1.,neuron_type="relu",reg_type="L2",verbose=false,init_type="zero",output_neuron_type="tanh",verbose=false)
criticnn = buildNN([2;10;10;1],0.001,-5000.,100.,neuron_type="relu",reg_type="L2",verbose=false,init_type="rand",output_neuron_type="none",verbose=false)

actor = NNDPGActor(actornn,[1.],[-1.],[1.],[-1.],0.0001)
critic = NNDPGCritic(criticnn,zeros(length(squash(criticnn,criticnn.weights,criticnn.biases))),[1.],[-1.])
param = NNDPGParam(100,0.0001)
solver = NNDPG.Solver()
updater = NNDPG.DPGUpdater()
=#

actor,critic,param, solver,updater = easyInit(2,[1.],[-1.],100,0.0001,
                  cw=0.,cth=0.,ActorLayers=[5.],CriticLayers=[5.],mu=0.9)

alpha_th = 0.000001
alpha_v = 0.0001
alpha_w = 0.0001

#println(selectAction(gm,actor,critic,p,s,s0))
#println(predict(actor.nn,s0))
#println(size(predict(actor.nn,s0)))
#println(gradient(actor,s0))
println(methods(train))

trainRNG = MersenneTwister(100)
simRNG = MersenneTwister(1000)
actRNG = MersenneTwister(500)

policy, hist = NNDPG.train(gm,trainRNG,actor, critic,param,solver,updater,
               time_horizon=5000,num_episodes=10,eps = 0.5,alpha=[alpha_th; alpha_w;alpha_v],gamma=0.99,natural=true, verbose=true,eps=0.25)

#physics inspired (optimal?) heuristic
#policy = (s)->s[2]>0.?1.:-1.


println(methods(runSim))
nSims = 100
R_,hists = NNDPG.runSim(gm,simRNG,actRNG,policy,time_horizon=5000,recordHist=false,nSims=nSims,verbose=true)

