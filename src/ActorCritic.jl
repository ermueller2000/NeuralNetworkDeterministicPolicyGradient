#ActorCritic.jl


#TODO: switch over to a type Union of Ints and Float--you wouldn't be doing policy gradient if you had nebulous states


module ActorCritic

export GenerativeModel, Actor, Critic, Param, train, Solver, Updater, simulate, runSim, History

abstract Actor
abstract Critic
abstract Param
abstract Updater


type GenerativeModel
  initState::Function
  nextState::Function
  isEnd::Function
  reward::Function
end

type Solver
  selectAction::Function
  reset!::Function
  updateWeights!::Function
end

#Solver() = Solver(selectAction,reset!,updateWeights!)

function train(gm::GenerativeModel,rng::AbstractRNG,actor::Actor, critic::Critic,p::Param, solver::Solver,u::Updater;
               time_horizon::Int=20,num_episodes::Int=10,eps::Float64 = 0.5,alpha::Array{Float64,1}=[0.01],gamma::Float64=0.99,natural::Bool=false,
               verbose::Bool=false,minibatch_size::Int=1,experience_replay::Bool=true)

  q = zeros(num_episodes)

  println()
  str = ""
  for episode = 1:num_episodes
    Q=0.
    t = 0
    if verbose
      print(repeat("\b \b",length(str)))
      str = "Training Episode $episode"
      print(str)
    end
    s = gm.initState(rng)
    solver.reset!(gm,actor,critic,p,s)
    a = solver.selectAction(gm,rng,actor,critic,p,s,eps)
    for t = 1:time_horizon
      r = gm.reward(s,a)
      s_ = gm.nextState(rng,s,a)
      endflag = gm.isEnd(s)
      if t == time_horizon
        endflag = true
      end
      Q += solver.updateWeights!(gm,actor,critic,p,solver,u,s,a,r,s_,endflag,alpha,gamma,natural,experience_replay,minibatch_size)
      if endflag
        #update statistics
        break
      end
      s = s_

      a = solver.selectAction(gm,rng,actor,critic,p,s,eps)
    end #t
    #sample state space to est. avg Q-value?
    q[episode] = Q./t
  end#episodes

  return (rng,s)->solver.selectAction(gm,rng,actor,critic,p,s),q
end

function simulate(gm::GenerativeModel,simRNG::AbstractRNG,actRNG::AbstractRNG,policy::Function;time_horizon::Int=20,recordHist::Bool=false)

  #Init state, action
  s = gm.initState(simRNG)
  a = policy(actRNG,s)
  #Init histories
  histX = typeof(s)[]
  histU = typeof(a)[]
  push!(histX,s)
  push!(histU,a)

  R = 0.
  for t = 1:time_horizon
    R += gm.reward(s,a)
    s = gm.nextState(simRNG,s,a)
    a = policy(actRNG,s)
    if recordHist
      push!(histX,s)
      push!(histU,a)
    end
    if gm.isEnd(s)
      break
    end
  end

  hist = History{typeof(s),typeof(a)}(histX,histU)

  return R, hist
end

function runSim(gm::GenerativeModel,simRNG::AbstractRNG,actRNG::AbstractRNG,policy::Function;time_horizon::Int=20,recordHist::Bool=false,nSims::Int=100,verbose::Bool=false)
  #just a function that runs a bunch of simulations and aggregates results
  println()
  str = ""
  R = 0.
  histories = Dict{Int,History}()
  for i = 1:nSims
    r, hist = simulate(gm,simRNG,actRNG,policy,time_horizon=time_horizon,recordHist=recordHist)
    R += r
    #push!(histories,hist)
    histories[i] = hist
    if verbose
      print(repeat("\b \b",length(str)))
      str = "Sim $i"
      print(str)
    end
  end

  if verbose
    println()
    println(R./nSims)
  end

  return (R./nSims), histories
end

type History{S,T}
  histX::Array{S,1}
  histU::Array{T,1}
end

################################
##placeholder functions#######
function selectAction(rng::AbstractRNG,gm::GenerativeModel,actor::Actor,critic::Critic,p::Param,s,eps::Float64=0.)
  error("Abstract \"Actor\", \"Critic\" not implemented!")
end

function updateWeights!(gm::GenerativeModel, actor::Actor,critic::Critic,p::Param,s,a,r,s_,endflag::Bool;
                        alpha::Array{Float64,2}=[0.01],gamma::Float64=0.99)
  error("Abstract \"Actor\", \"Critic\" not implemented!")
end

function reset!(gm::GenerativeModel,actor::Actor,critic::Critic,p::Param,s)
  error("Abstract \"Actor\", \"Critic\" not implemented!")
end


end #module
