#DPG.jl

module DPG

using ActorCritic
import ActorCritic.Solver, ActorCritic.GenerativeModel, ActorCritic.train

export DPGCritic, DPGActor, DPGParam, updateWeights!, Experience, Solver, GenerativeModel, train, DPGUpdater, simulate, runSim

##########################################################################################################
#####For a Generic linear deterministic policy gradient actor critic--this is the prototype################
#NOTE: critic should be updated more quickly than the actor
abstract DPGCritic <: Critic
abstract DPGActor <: Actor
abstract DPGParam <: Param

type DPGUpdater <: Updater
  getValue::Function
  gradient::Function
  actorUpdate!::Function
  criticUpdate!::Function
end

type Experience
  s::Array{Float64,1}
  a::Array{Float64,1}
  r::Float64
  s_::Array{Float64,1}
end

#Solver() = Solver(selectAction,reset!,updateWeights!)

###Placeholder functions
function getValue(critic::DPGCritic,s)
  error("Abstract \"DPGCritic\" not implemented")
end
function getValue(critic::DPGCritic,s,next::Bool)
  error("Abstract \"DPGCritic\" not implemented")
end
function gradient(actor::DPGActor,s)
  error("Abstract \"DPGActor\" not implemented")
end
function selectAction(gm::GenerativeModel,rng::AbstractRNG,actor::DPGActor,critic::DPGCritic,p::DPGParam,s,eps::Float64=0.)
  error("Abstract \"DPGCritic\" and \"DPGActor\" not implemented")
end
function selectAction(gm::GenerativeModel,actor::DPGActor,critic::DPGCritic,p::DPGParam,s,next::Bool)
  error("Abstract \"DPGCritic\" and \"DPGActor\" not implemented")
end
function actorUpdate!(actor::DPGActor,critic::DPGCritic,gradient::Array{Float64,2},alpha::Float64)
  error("Abstract \"DPGCritic\" and \"DPGActor\" not implemented")
end
function criticUpdate!(critic::DPGCritic,target::Float64,advantage::Float64,estimate::Float64,s,alpha::Float64)
  error("Abstract \"DPGCritic\" not implemented")
end

function reset!(gm::GenerativeModel,actor::DPGActor,critic::DPGCritic,p::DPGParam,s)
  error("Abstract \"DPGActor\", \"DPGCritic\" not implemented!")
end

function updateWeights!(gm::GenerativeModel,actor::DPGActor,critic::DPGCritic,p::Param,solver::Solver,u::DPGUpdater,
                        s,a,r,s_,endflag::Bool,
                        alpha::Array{Float64,1},gamma::Float64,natural::Bool=false,experience_replay::Bool=false,minibatch_size::Int=1)
  assert(length(alpha) == 3)
  if experience_replay
    if length(p.memory) >= p.capacity
      #remove element from memory
      splice!(p.memory,rand(1:length(p.memory)))
    else
      #dont do updates until memory is full?
      #return
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

    u.actorUpdate!(actor,critic,J,alpha[1],natural)
    if !endflag
      u.criticUpdate!(critic,r+gamma*Q_,A[1],Q,s,alpha[3])
    else
      u.criticUpdate!(critic,r,A[1],Q,s,alpha[3])
    end

    critic.w += alpha[2]*(del*phi_sa - p.cw*critic.w) #<-regularization term
    #doesn't need a separate function because this has have this form because of compatible function blah blah blah
    vs[i] = v
    if experience_replay
      mem = p.memory[rand(1:length(p.memory))]
      s = mem.s
      a = mem.a
      r = mem.r
      s_ = mem.s_
    end
  end
  return mean(vs)#norm(critic.w)

end
######

end #module
