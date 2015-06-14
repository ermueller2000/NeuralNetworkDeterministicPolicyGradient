#MTNCAR.jl


module MTNCAR

export nextState, reward, isEnd, init, actions, reward2

function init()
  s0 = [-0.5;0.0]
  return s0
end

function nextState(s::Array{Float64,1},u::Float64)
  x = s[1]
  v = s[2]
  v_ = v + u*0.001+cos(3*x)*-0.0025
  v_ = max(min(0.07,v_),-0.07)
  x_ = x+v_
  #inelastic boundary
  if x_ < -1.2
    x_ = -1.2
    v_ = 0.
  end

  return [x_;v_]
end

function nextState(s::Array{Float64,1},u::Array{Float64,1})
  return nextState(s,u[1])
end

function reward(s::Array{Float64,1},u::Float64)
  return -1.#-sin(pi/2+0.5+s[1])#-1.
  #or the height version
end

function reward(s::Array{Float64,1},u::Array{Float64,1})
  return -1.#-sin(pi/2+0.5+s[1])#-1.
  #or the height version
end

function reward2(s::Array{Float64,1},u::Array{Float64,1})
  return -sin(pi/2+0.5+s[1])
  #or the height version
end

function reward2(s::Array{Float64,1},u::Float64)
  return -sin(pi/2+0.5+s[1])
  #or the height version
end

function isEnd(s::Array{Float64,1})
  x = s[1]
  v = s[2]
  return x >= 0.5
end

function actions(s::Array{Float64,1})
  return [-1.;0.;1.]
end


end #module
