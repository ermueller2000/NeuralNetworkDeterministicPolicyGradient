#NNaux.jl
#Type definitions and useful functions
#TODO: define -?

#put into NNaux.jl?
#########Types and associated utility functions ########
########################################################
#how to make the size of the array modular?
type NNWeights{d}
  x::Dict{Int,Array{Float64,d}}
end

type NeuralNetwork
  #size::Array{Int,1}
  #weights::Array{Float64,2}#dict layer number ->array{float,1}?
  weights::NNWeights
  #biases::Array{Float64,2}
  biases::NNWeights
  neuron::Function
  gradient::Function
  output_neuron::Function
  output_gradient::Function
  loss_function::Function
  loss_gradient::Function
  c::Float64 #regularization term
  reg::String #type of regularization

  #lam1::Float64 #L1 regularization term
  #lam2::Float64 #L2 regularization term
  #alternatively, have a arbitrary "c" parameter that gets related to one of the regularization types: L1, L2, maxnorm, dropout
  yu::Array{Float64,1}
  yl::Array{Float64,1}
  xu::Array{Float64,1}
  xs::Array{Float64,1}
  verbose::Bool
  scale::Bool
end

function getindex(x::NNWeights,i::Int)
  return x.x[i]
end

function setindex!(x::NNWeights,y::Array{Float64,2},i::Int)
  x.x[i] = y
end

function setindex!(x::NNWeights,y::Array{Float64,1},i::Int)
  x.x[i] = y
end

function length(x::NNWeights)
  return length(x.x)
end

function add!(x::NNWeights,y::NNWeights)
  #check this vvvv
  L = length(x.x)
  assert(L == length(y.x))
  #assert keys are equal?
  assert(length(setdiff(keys(x.x),keys(y.x))) == 0)

  for i = 1:L
    x.x[i] += y.x[i]
  end

  return x
end

function *(x::NNWeights,a::Real)
  y = zero(x)
  L = length(x.x)
  for i = 1:L
    y.x[i] = x.x[i]*a
  end
  return y
end

function *(a::Real,x::NNWeights)
  return x*a
end

function +(x::NNWeights, y::NNWeights)
  L = length(x.x)
  assert(L == length(y.x))
  #assert keys are equal? Do I even need this assertion?--does it even make that much perf difference?
  assert(length(setdiff(keys(x.x),keys(y.x))) == 0)
  z = zero(x)
  for i = 1:L
    z.x[i] = x.x[i] + y.x[i]
  end
  return z
end

function +(x::NNWeights,y::Float64)
  return x + one(x)*y
end

function +(y::Float64,x::NNWeights)
  return x + y
end

function mult!(x::NNWeights,a::Real)
  L = length(x.x)
  for i = 1:L
    x.x[i] *= a
  end
  return x
end

function zero(x::NNWeights)
  #make a zero NNweight of the same size
  L = length(x.x)
  d = length(size(x.x[1]))
  y = Dict{Int,Array{Float64,d}}([i=>zeros(size(x.x[i])) for i = 1:L])
  return NNWeights{d}(y)
end

function zero(sizes)
  #i want the type of the parameter to be an array of integer tuples of length 1 or 2
  L = length(sizes)
  d = length(sizes[1])
  x = Dict{Int,Array{Float64,d}}([i=>zeros(sizes[i]...) for i = 1:L])
  return NNWeights{d}(x)
end


function one(x::NNWeights)
  #make a zero NNweight of the same size
  L = length(x.x)
  d = length(size(x.x[1]))
  y = Dict{Int,Array{Float64,d}}([i=>ones(size(x.x[i])) for i = 1:L])
  return NNWeights{d}(y)
end

function one(sizes)
  #i want the type of the parameter to be an array of integer tuples of length 1 or 2
  L = length(sizes)
  d = length(sizes[1])
  x = Dict{Int,Array{Float64,d}}([i=>ones(sizes[i]...) for i = 1:L])
  return NNWeights{d}(x)
end

function randweights(sizes)
  #i want the type of the parameter to be an array of integer tuples of length 1 or 2
  #only for initialization of parameters
  L = length(sizes)
  d = length(sizes[1])
  x = Dict{Int,Array{Float64,d}}([i=>(rand(sizes[i]...)-0.5)./sqrt(2.0/sizes[i][2]) for i = 1:L])
  return NNWeights{d}(x)
end

function norm(x::NNWeights)
  L = length(x.x)
  val = 0
  for i = 1:L
    val += sum([j*j for j in x.x[i][:]])#vecnorm(x.x[i])
  end
  return val
end

function ^(x::NNWeights,i::Int)
  y = zero(x)
  L = length(x.x)
  for i = 1:L
    y.x[i] = x.x[i].*x.x[i]
  end
  return y
end

function /(x::NNWeights,y::NNWeights)
  z = zero(x)
  L = length(x.x)
  for i = 1:L
    z.x[i] = x.x[i]./y.x[i]
  end
  return z
end

function sqrt(x::NNWeights)
  y = zero(x)
  L = length(x.x)
  for i = 1:L
    y.x[i] = sqrt(x.x[i])
  end
  return y
end
#######################################################################
