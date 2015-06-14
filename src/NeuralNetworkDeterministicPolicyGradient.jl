include("ActorCritic.jl")
include("DPG.jl")
include("NN.jl")

module NeuralNetworkDeterministicPolicyGradient

using DPG, NN

import DPG.GenerativeModel, DPG.Solver, DPG.train, DPG.updateWeights!, DPG.DPGUpdater,DPG.Updater
export NNDPGActor, NNDPGCritic, NNDPGParam, Solver,DPGUpdater, reset!, GenerativeModel, gradient, getZeroIndices, simulate, runSim, easyInit,train

Solver() = Solver(selectAction,reset!,updateWeights!)
Solver(s::String) = (s=="batchrmsprop")?Solver(selectAction,reset!,batchUpdateWeights!):error("no such solver $s")
DPGUpdater() = DPGUpdater(getValue,gradient,actorUpdate!,criticUpdate!)
DPGUpdater(s::String) = (s=="rmsprop")?DPGUpdater(getValue,gradient,RMSPropActorUpdate!,RMSPropCriticUpdate!):
          ((s=="batchrmsprop")?DPGUpdater(getValue,gradient,BatchRMSPropActorUpdate!,RMSPropCriticUpdate!):error("no such updater $s"))


function easyInit(n::Int,ub::Array{Float64,1},lb::Array{Float64,1},memory_size::Int,cv::Float64;
                  cw::Float64=0.,cth::Float64=0.,ActorLayers::Array{Float64,1}=[2.],CriticLayers::Array{Float64,1}=[2.],mu::Float64=0.9,neuron_type::String="relu")
  if false in (ub .> lb)
    error("inconsistent bounds")
  end
  nA = length(ub)
  actorNN = buildNN([n;ifloor(n*ActorLayers);nA],0.01,lb,ub,
                    neuron_type=neuron_type,reg_type="L2",verbose=false,init_type="rand",output_neuron_type="tanh",scale=false)
  criticNN = buildNN([n;ifloor(n*CriticLayers);1],cv,typemin(Float64),typemax(Float64),
                     neuron_type=neuron_type,reg_type="L2",verbose=false,init_type="rand",output_neuron_type="none",scale=false)

  nth = length(squash(actorNN))

  actor = NNDPGActor(actorNN,ub,lb,ones(n),-1*ones(n),cth,mu)
  critic = NNDPGCritic(criticNN,(rand(nth)-0.5)*0.1,ones(n),-1*ones(n))
  param = NNDPGParam(memory_size,cw)
  solver = Solver("batchrmsprop")
  updater = DPGUpdater("batchrmsprop")

  return actor, critic, param, solver, updater

end


type NNDPGActor <: DPGActor
  nn::NeuralNetwork
  ub::Array{Float64,1}
  lb::Array{Float64,1}
  xu::Array{Float64,1}
  xl::Array{Float64,1}
  cth::Float64
  mu::Float64 #exp decay rate for rmsprop
  vW::NNWeights
  vb::NNWeights
  NNDPGActor(nn,ub,lb,xu,xl,cth,mu) = (false in (xu .> xl))||(false in (ub .> lb))? error("incorrect bounds"): new(nn,ub,lb,xu,xl,cth,mu,zero(nn.weights),zero(nn.biases))
  NNDPGActor(nn,ub,lb,xu,xl,cth) = (false in (xu .> xl))||(false in (ub .> lb))? error("incorrect bounds"): new(nn,ub,lb,xu,xl,cth,0.9,zero(nn.weights),zero(nn.biases))
end

type NNDPGCritic <: DPGCritic
  nn::NeuralNetwork
  w::Array{Float64,1}
  xu::Array{Float64,1}
  xl::Array{Float64,1}
  mu::Float64 #exp decay rate for rmsprop
  vW::NNWeights
  vb::NNWeights
  NNDPGCritic(nn,w,xu,xl,mu) = (false in (xu .> xl))? error("incorrect bounds"): new(nn,w,xu,xl,mu,zero(nn.weights),zero(nn.biases))
  NNDPGCritic(nn,w,xu,xl) = (false in (xu .> xl))? error("incorrect bounds"): new(nn,w,xu,xl,0.9,zero(nn.weights),zero(nn.biases))
end

type NNDPGParam <: DPGParam
  capacity::Int
  memory::Array{Experience,1}
  cw::Float64
  NNDPGParam(x,cw) = new(x,Experience[],cw)
end

function scale_s(s,xu::Array{Float64,1},xl::Array{Float64,1})
  ####I think issue is here
  if length(size(s)) == 2
    m = size(s,2)
    XU = repmat(xu,1,m)
    XL = repmat(xl,1,m)
    return (s - (XU+XL)./2)./((XU-XL)./2)
  end
  return (s - (xu+xl)./2)./((xu-xl)./2)
end

function reset!(gm::GenerativeModel,actor::NNDPGActor,critic::NNDPGCritic,p::NNDPGParam,s)
  #does nothing--would clear out the neurons for RNN/LSTM
end

function getValue(critic::NNDPGCritic,s,next::Bool=false)
  return predict(critic.nn,scale_s(s,critic.xu,critic.xl))[1]
end

function selectAction(gm::GenerativeModel,rng::AbstractRNG,actor::NNDPGActor,critic::NNDPGCritic,p::NNDPGParam,s,eps::Float64=0.)
  mu = predict(actor.nn,scale_s(s,actor.xu,actor.xl))
  noise = (actor.ub-actor.lb).*eps.*randn(rng,length(mu))
  #vector-wise min/max to get things bounded
  a = mu + noise
  return vec(minimum(hcat(actor.ub,maximum(hcat(actor.lb,a),2)),2))
end

function selectAction(gm::GenerativeModel,actor::NNDPGActor,critic::NNDPGCritic,p::NNDPGParam,s,next::Bool)
  mu = predict(actor.nn,scale_s(s,actor.xu,actor.xl))
  #vector-wise min/max to get things bounded
  return vec(minimum(hcat(actor.ub,maximum(hcat(actor.lb,mu),2)),2))
end

#have different version of updateWeights where we use experience replay

#only reason you'd be using a NN + DPG is if you had continuous valued states and actions

function criticUpdate!(critic::NNDPGCritic,target::Float64,advantage::Float64,estimate::Float64,s,alpha::Float64)
  #error("not implemented")
  NN.train(scale_s(s,critic.xu,critic.xl),target-advantage,critic.nn,alpha=alpha,calculate_loss=false,num_epochs=1,update_type="SGD")
end

function actorUpdate!(actor::NNDPGActor,critic::NNDPGCritic,gradient::Array{Float64,2},alpha::Float64,natural::Bool=false)
  if natural
    dth = alpha*(critic.w - actor.cth*squash(actor.nn,actor.nn.weights,actor.nn.biases))
  else
    dth = alpha*(gradient*(gradient'*critic.w) - actor.cth*squash(actor.nn,actor.nn.weights,actor.nn.biases))
  end
  dW, db = desquash(actor.nn,dth)
  actor.nn.weights += dW
  actor.nn.biases += db
end

function RMSPropCriticUpdate!(critic::NNDPGCritic,target::Union(Array{Float64,2},Array{Float64,1},Float64),
                              advantage::Union(Array{Float64,2},Array{Float64,1},Float64),
                              estimate::Union(Array{Float64,2},Array{Float64,1},Float64),s,alpha::Float64)
  #error("not implemented")
  #vW, vb should be updated in place
  NN.train(scale_s(s,critic.xu,critic.xl),target-advantage,critic.nn,critic.vW,critic.vb,alpha=alpha,mu=critic.mu,calculate_loss=false,num_epochs=1,update_type="rmsprop")
end

function RMSPropActorUpdate!(actor::NNDPGActor,critic::NNDPGCritic,gradient::Array{Float64,2},alpha::Float64,natural::Bool=false)
  if natural
    dth = (critic.w - actor.cth*squash(actor.nn,actor.nn.weights,actor.nn.biases))
  else
    dth = (gradient*(gradient'*critic.w) - actor.cth*squash(actor.nn,actor.nn.weights,actor.nn.biases))
  end
  dW, db = desquash(actor.nn,dth)

  actor.vW = actor.mu*actor.vW + (1-actor.mu)*(dW)^2
  actor.vb = actor.mu*actor.vb + (1-actor.mu)*(db)^2

  actor.nn.weights += (alpha)*dW/sqrt(actor.vW+1e-8)
  actor.nn.biases += (alpha)*db/sqrt(actor.vb+1e-8)

end

function BatchRMSPropActorUpdate!(actor::NNDPGActor,critic::NNDPGCritic,gradient::Array{Float64,2},alpha::Float64,natural::Bool=false)
  if natural
    dth = (critic.w - actor.cth*squash(actor.nn,actor.nn.weights,actor.nn.biases))
  else
    dth = (gradient*(gradient'*critic.w) - actor.cth*squash(actor.nn,actor.nn.weights,actor.nn.biases))
  end
  dW, db = desquash(actor.nn,dth)

  return dW, db

end


function gradient(actor::NNDPGActor,s)
  #you do a forward pass, and a modified backprop pass, (no y-y_ term), and squash it into a vector
  #error("Not implemented")
  #####The code for the analytical gradient--needs to be checked
  y,a,z = feedforward(scale_s(s,actor.xu,actor.xl),actor.nn)
  dW,db = backprop(a,z,y-1.,actor.nn)
  dth = squash(actor.nn,dW,db)
  #some kind of something to remove the terms in the last layer that don't matter for each action, and hcat these vectors

  J = zeros(length(dth),length(y))
  for i = 1:length(y)
    dth_ = deepcopy(dth)
    dth_[getZeroIndices(actor.nn,i)] = 0.
    J[:,i] = dth_
  end

  return J
end

function numericalGradient(actor::NNDPGActor,s)
  #numerical gradient
  W,b = actor.nn.weights, actor.nn.biases
  eps = 0.0001
  w = squash(actor.nn,W,b)
  g_ = zeros(length(w))
  a1 = []
  for ind = 1:length(w)
  #ind = rand(1:length(w))
      w1 = w[:]
      w1[ind] += eps
      w2 = w[:]
      w2[ind] -= eps

      W1,b1 = desquash(actor.nn,w1)
      W2,b2 = desquash(actor.nn,w2)
      nn1 = deepcopy(actor.nn)
      nn1.weights = W1
      nn1.biases = b1
      nn2 = deepcopy(actor.nn)
      nn2.weights = W2
      nn2.biases = b2

      a1 = predict(nn1,scale_s(s,actor.xu,actor.xl))[1]
      a2 = predict(nn2,scale_s(s,actor.xu,actor.xl))[1]
      g_[ind] = (a1-a2)./(2*eps)
  end
  dth = g_
  y = a1

  #this part stays the same
  J = zeros(length(dth),length(y))
  for i = 1:length(y)
    dth_ = deepcopy(dth)
    dth_[getZeroIndices(actor.nn,i)] = 0.
    J[:,i] = dth_
  end

  return J
end

function getZeroIndices(nn::NeuralNetwork,i::Int)
  #a helper function to say which indices of a squashed vector form of the derivatives are zero when taken wrt to output i
  L = length(nn.weights.x)

  candidates = [1:(length(nn.weights.x[L])+length(nn.biases.x[L]))]

  h,w = size(nn.weights.x[L])

  safe = [i+(j-1)*h for j = 0:w+1]

  return setdiff(candidates,safe)

end


function batchUpdateWeights!(gm::GenerativeModel,actor::DPGActor,critic::DPGCritic,p::DPGParam,solver::Solver,u::DPGUpdater,
                        s,a,r,s_,endflag::Bool,
                        alpha::Array{Float64,1},gamma::Float64,natural::Bool=false,experience_replay::Bool=false,minibatch_size::Int=1)
  assert(length(alpha) == 3)

  #For batching: aggregate target, state for criticupdate,

  if experience_replay
    if length(p.memory) >= p.capacity
      #remove element from memory
      splice!(p.memory,rand(1:length(p.memory)))
    else
      #dont do updates until memory is full?
      #return
      push!(p.memory,Experience(s,a,r,s_))
      return 0.
    end
    push!(p.memory,Experience(s,a,r,s_))
    mem = p.memory[rand(1:length(p.memory))]
    s = mem.s
    a = mem.a
    r = mem.r
    s_ = mem.s_
    num_iters = minibatch_size
  else
    num_iters = 1
  end
  vs = zeros(num_iters)
  batch_s = Union(Float64,Int)[]
  batch_target = zeros(num_iters)
  batch_phi_sa = Float64[]
  batch_advantage = zeros(num_iters)
  batch_estimate = zeros(num_iters)
  dW = zero(actor.nn.weights)
  db = zero(actor.nn.biases)


  #I /think/ a for loop is equivalent to minibatching
  for i = 1:num_iters
    #first for th, second for w, third for v
    #COPDAC-Q
    #order matters here!!! vvv
    v_ = u.getValue(critic,s_,true) #advantage function/regret is 0 since we are assuming we're following the actor policy
    v = u.getValue(critic,s)

    J = u.gradient(actor,s) #a 2d array matrix
    mu = solver.selectAction(gm,actor,critic,p,s,true) #special case
    phi_sa= J*(a-mu) #w is probably associated with some actor/critic object
    A = phi_sa'*critic.w
    Q = A[1] + v
    Q_ = v_
    del = r + gamma*Q_ - Q


    dW_, db_ = u.actorUpdate!(actor,critic,J,alpha[1],natural)

    dW += dW_
    db += db_

    if i == 1
      batch_s = repmat(s,1,1)
      batch_phi_sa = del*phi_sa
    else
      batch_s = hcat(batch_s,s)
      batch_phi_sa += del*phi_sa
    end

    batch_advantage[i] = A[1]

    batch_estimate[i] = Q

    if !endflag
      batch_target[i] = r+gamma*Q_
    else
      batch_target[i] = r
    end



    #doesn't need a separate function because this has have this form because of compatible function blah blah blah
    vs[i] = v
    #u.actorUpdate!(actor,critic,J,alpha[1],natural)
    #u.criticUpdate!(critic,r+gamma*Q_,Q,s,alpha[3])
    if experience_replay
      mem = p.memory[rand(1:length(p.memory))]
      s = mem.s
      a = mem.a
      r = mem.r
      s_ = mem.s_
    end
  end


  critic.w += alpha[2]*(batch_phi_sa./num_iters - p.cw*critic.w) #<-regularization term

  actor.vW = actor.mu*actor.vW + (1-actor.mu)*(dW)^2
  actor.vb = actor.mu*actor.vb + (1-actor.mu)*(db)^2

  actor.nn.weights += (alpha[1]/num_iters)*dW/sqrt(actor.vW+1e-8)
  actor.nn.biases += (alpha[1]/num_iters)*db/sqrt(actor.vb+1e-8)

  u.criticUpdate!(critic,batch_target,batch_advantage,batch_estimate,batch_s,alpha[3])

  return mean(vs)#norm(critic.w)

end


end # module
