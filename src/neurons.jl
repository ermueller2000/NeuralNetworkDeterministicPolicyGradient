#neurons.jl
#for use with NN.jl and NNaux.jl
#really just here to keep things neat

########Activation Functions and Gradients ######################
#################################################################
#activation functions and gradients--might put in a different file to keep things neat
function sigm(x)
  return 1./(1+exp(-x))
end

function reLu(x)
  return max(0.,x)
end

function leakyReLu(x,alpha::Float64=0.01)
  return max(alpha*x,x)
end

function softplus(x)
  return log(1+exp(x))
end

#gradients/derivatives of activation functions--elementwise
function Gsigm(x)
  return sigm(x).*(1-sigm(x))
end

function GreLu(x)
  return x > 0.? 1.: 0.
end

function GleakyReLu(x,alpha::Float64=0.01)
  return x > 0.? 1.: alpha
end

function Gtanh(x)
  return 1 - tanh(x).*tanh(x)
end

#=
function Gsoftplus(x::real)
  return sigm(x)
end

@vectorize_1arg Real GSoftplus
=#

@vectorize_1arg Real sigm
@vectorize_1arg Real reLu
@vectorize_1arg Real softplus
@vectorize_1arg Real Gsigm
@vectorize_1arg Real Gtanh
@vectorize_1arg Real GreLu
@vectorize_1arg Real GleakyReLu



#############################################################
####Loss functions###
#they don't quite belong here, but it makes slightly more sense than sticking it in NNaux.jl

function L1Loss(y,y_)
  return norm(y-y_,1)
end

function L1Gradient(y,y_)
  return sign(y-y_)
end

function L2Loss(y,y_)
  return (y-y_)'*(y-y_)
end

function L2Gradient(y,y_)
  return y-y_
end

@vectorize_2arg Real L1Loss
@vectorize_2arg Real L2Loss
@vectorize_2arg Real L1Gradient
@vectorize_2arg Real L2Gradient
