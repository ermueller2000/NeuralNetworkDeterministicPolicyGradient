#NN.jl
#replace with: https://github.com/pluskid/Mocha.jl

#Working!!!!
#TODO: switch from having those extra functions to using type unions
#TODO: gradient checking, adagrad etc...
#TODO: dropout regularization, maxnorm regularization, maxout neurons
#TODO: figure out nice way to do learning rate annealing, changing the loss function: softmax?, huber!?!?
#TODO: only update some things for missing y data entries (0 out output gradient?)
#TODO: pretrain stuff, inputs that result in max activations as opt prob
#TODO: code profiling
#TODO: LSTM/RNN stuff!!!
#NOTE: can probably reduce parameter size Float64->Float32 because papers
#NOTE: might be more efficient to have explicit Integer and Float versions rather than casting everything as real
#NOTE: yu, yl might be deprecated--nvm only use with sigm/tanh units

module NN

import Base.length, Base.norm, Base.sqrt
export buildNN, train, predict, squash, desquash, feedforward, backprop, NeuralNetwork,zero,+,*,/,^,sqrt,NNWeights #, pretrain

include("NNaux.jl")
include("neurons.jl")

function buildNN(sizes,c::Float64,yl::Array{Float64,1},yu::Array{Float64,1},loss_function::Function=L2Loss,loss_gradient::Function=L2Gradient;
                 neuron_type::String="reLu",reg_type::String="L2",verbose::Bool=True,init_type::String="rand",output_neuron_type::String="none",scale::Bool=true)
  #initialize weights and build neural network type based on these inputs
  if verbose
    println("Initializing Neural Network")
  end

  if lowercase(neuron_type) == "relu"
    neuron = reLu
    gradient = GreLu
  elseif lowercase(neuron_type) == "leakyrelu"
    neuron = leakyReLu
    gradient = GleakyReLu
  elseif lowercase(neuron_type) == "tanh"
    neuron = tanh
    gradient = Gtanh
  elseif lowercase(neuron_type) == "sigm"
    neuron = sigm
    gradient = Gsigm
  elseif lowercase(neuron_type) == "softplus"
    neuron = softplus
    gradient = Gsoftplus
    error("Gradient not working")
  else
    error(string("Undefined neuron type ",neron_type))
  end


  if lowercase(output_neuron_type) == "relu"
    output_neuron = reLu
    output_neuron_gradient = GreLu
  elseif lowercase(output_neuron_type) == "leakyrelu"
    output_neuron = leakyReLu
    output_neuron_gradient = GleakyReLu
  elseif lowercase(output_neuron_type) == "tanh"
    output_neuron = tanh
    output_neuron_gradient = Gtanh
  elseif lowercase(output_neuron_type) == "sigm"
    output_neuron = sigm
    output_neuron_gradient = Gsigm
  elseif lowercase(output_neuron_type) == "softplus"
    output_neuron = softplus
    output_neuron_gradient = Gsoftplus
    error("Gradient not working")
  elseif lowercase(output_neuron_type) == "none"
    output_neuron = (x)->x
    output_neuron_gradient = (x)->ones(size(x))
  else
    error(string("Undefined neuron type ",neron_type))
  end

  assert(length(yl) == length(yu))
  assert(!( false in (yl.<yu) ) )

  #assume full connectivity
  sizes_ = [(sizes[i+1],sizes[i]) for i = 1:length(sizes)-1]

  if lowercase(init_type) == "rand"
    weights = randweights(sizes_)
  elseif lowercase(init_type) == "zero"
    weights= zero(sizes_)
  else
    error(string("no such weight initialization method: ", init_type))
  end
  biases = zero(sizes[2:end])


  nn = NeuralNetwork(weights, biases, neuron, gradient,output_neuron,output_neuron_gradient, loss_function, loss_gradient,
                     c, reg_type, yu, yl,zeros(sizes[1]),ones(sizes[1]), verbose,scale)

  return nn
  #maybe do some pretraining with stacked denoising autoencoders?
end

#i'm sure there's a macro to make this easier

function buildNN(sizes,c::Float64,yl::Float64,yu::Float64,loss_function::Function=L2Loss,loss_gradient::Function=L2Gradient;
                 neuron_type::String="reLu",reg_type::String="L2",verbose::Bool=True,init_type::String="zero",output_neuron_type::String="none",scale::Bool=true)
  return buildNN(sizes,c,[yl],[yu],loss_function,loss_gradient,
                 neuron_type=neuron_type,reg_type=reg_type,verbose=verbose,init_type=init_type,output_neuron_type=output_neuron_type,scale=scale)
end

function buildNN(sizes,c::Float64,yl::Float64,yu::Float64,loss_type::String="L2";
                 neuron_type::String="reLu",reg_type::String="L2",verbose::Bool=True,init_type::String="zero",output_neuron_type::String="none",scale::Bool=true)

  return buildNN(sizes,c,[yl],[yu],loss_type,
                 neuron_type=neuron_type,reg_type=reg_type,verbose=verbose,init_type=init_type,output_neuron_type=output_neuron_type,scale=scale)
end

function buildNN(sizes,c::Float64,yl::Array{Float64,1},yu::Array{Float64,1},loss_type::String="L2";
                 neuron_type::String="reLu",reg_type::String="L2",verbose::Bool=True,init_type::String="zero",output_neuron_type::String="none",scale::Bool=true)

  if lowercase(loss_type) == "l2"
    loss_function = L2Loss
    loss_gradient = L2Gradient
  elseif lowercase(loss_type) == "l1"
    loss_function = L1Loss
    loss_gradient = L1Gradient
  else
    error(string("No such loss function type: ", loss_Type))
  end

  return buildNN(sizes,c,yl,yu,loss_function,loss_gradient,
                 neuron_type=neuron_type,reg_type=reg_type,verbose=verbose,init_type=init_type,output_neuron_type=output_neuron_type,scale=scale)
end


function pretrain(X::Array{Real,2},Y::Array{Real,2},nn::NeuralNetwork;
                  rho::Float64=0.05,alpha::Float64=0.01,beta::Float64=0.1,minibatch_size::Int=100,num_epochs::Int=10)
  ########################
  ###NOT IMPLEMENTED###
  #######################
  error("pretrain not implemented")
  #rho = sparsity parameter
  #beta = sparsity penalty

  #HOWTODO: for each layer, build a 2-layer NN, use normal feedforward, use AEBackProp
  #  BUT feedforward needs to be a sigm activation function on the end, so do pre/post scaling first
  #just going to do vanilla GD since we would have to compute rho_ for each update

  #pretraining might not be necessary if using relu/softplus units?
  if !(nn.neuron == sigm || nn.neuron == tanh)
    error("Can only pretrain with sigmoid or tanh units")
  end
  error("Not yet implemented")

  if nn.verbose
    println("Pretraining...")
  end
  #need data: X and Y
  #pretrain with AE or RBM?

  m = size(X,2)
  assert(m == size(Y,2))
  L = length(nn.biases)

  for l = 2:L-1
    #ignore input and output layers

    #something
    #pass through data to do SGD
    for epoch = 1:num_epochs
      for i = 1:m
        #minibatch GD stuff
        #for each example, need feed forward pass, and backprop pass
      end
    end #epoch

  end #l

  return nn
end

#maybe also have a train! function, probably not necessary though
#include vW, and vb as arguments and outputs so you can continue learning
function train(X::Array{Float64,2},Y::Array{Float64,2},nn::NeuralNetwork,vW::NNWeights=zero(nn.weights),vb::NNWeights=zero(nn.biases);
               alpha::Float64=0.01,mu::Float64=0.9,minibatch_size::Int=100,num_epochs::Int=10,
               update_type::String="momentum",calculate_loss::Bool=true,half_rate::Int=1,loss_fn::String="L1")
  #mu is the "mass" for nesterov/momentum, and it is also the decay rate for rmsprop
  #each column of X is a training example, same with Y--we do this to access things in memory order
  m = size(X,2)
  assert(m == size(Y,2))

  #whiten data
  if nn.scale
    nn.xu = vec(mean(X,2))
    nn.xs = vec(std(X,2))
  end
  losses = Float64[]
  losses_true = Float64[]

  L = length(nn.biases)

  dW = zero(nn.weights)
  db = zero(nn.biases)

  #becomes true gradient descent
  if m < minibatch_size
    minibatch_size = m
  end

  count = 0
  batch = 1
  s = ""
  for epoch = 1:num_epochs
    if nn.verbose
      println("Epoch $epoch")
    end
    if minibatch_size != m
      p = randperm(m)
    else
      p = [1:m]
    end
    for i_ = 1:m
      i = p[i_]
      x = (X[:,i]-nn.xu)./nn.xs
      y = Y[:,i]#(Y[:,i]-((nn.yl+nn.yu)/2))./((nn.yu-nn.yl)./2)
      y_, a, z = feedforward(x,nn)
      dW_, db_ = backprop(a,z,y,nn)


      dW += dW_
      db += db_
      #add!(dW,dW_)
      #add!(db,db_)

      #increment dW, db
      if mod(count,minibatch_size) == 0
        # fold all of this into a GDupdate function?
        #calculate loss function
        if calculate_loss
          loss = 0.
          for j = 1:m
            x = (X[:,j]-nn.xu)./nn.xs
            y = (Y[:,j])#-((nn.yl+nn.yu)/2))./((nn.yu-nn.yl)./2)
            y_, a, z = feedforward(x,nn)
            #loss += 0.5*norm(y-y_,1)#(y-y_)'*(y-y_)#squared error
            loss += 0.5*nn.loss_function(y,y_)
          end

          loss /= m

          losses = [losses;loss]

          #need some kind of loss term for here--this isn't accurate
          loss += 0.5*nn.c*norm(nn.weights)
          loss += 0.5*nn.c*norm(nn.biases)
          losses_true = [losses_true;loss]
        end

        if nn.verbose
          print(repeat("\b \b",length(s)))
          s = string("Minibatch: ",batch)
          print(s)
        end
        #update weights, biases (maybe include another function to make this cleaner)
        #might be a cleaner way to do this (make a function over weights than nn holds?)

        updateWeights!(nn, dW, db, vW, vb,alpha,mu,m,update_type)

        dW = zero(nn.weights)
        db = zero(nn.biases)

        #reset dW, db
        batch += 1

        if mod(batch,half_rate) == 0
          alpha /= 2
        end
      end

      count += 1
    end #i_
  end #epoch

  return nn, losses, losses_true, alpha, vW, vb

end

function train(X::Array{Float64,1},Y::Array{Float64,1},nn::NeuralNetwork,vW::NNWeights=zero(nn.weights),vb::NNWeights=zero(nn.biases);
               alpha::Float64=0.01,mu::Float64=0.9,minibatch_size::Int=100,num_epochs::Int=10,update_type::String="momentum",
               calculate_loss::Bool=true,half_rate::Int=1)
  return train(X',Y',nn,vW,vb,alpha=alpha,mu=mu,minibatch_size=minibatch_size,num_epochs=num_epochs,
               update_type=update_type,calculate_loss=calculate_loss,half_rate=half_rate)

end

function train(X::Array{Float64,2},Y::Array{Float64,1},nn::NeuralNetwork,vW::NNWeights=zero(nn.weights),vb::NNWeights=zero(nn.biases);
               alpha::Float64=0.01,mu::Float64=0.9,minibatch_size::Int=100,num_epochs::Int=10,update_type::String="momentum",
               calculate_loss::Bool=true,half_rate::Int=1)
  return train(X,Y',nn,vW,vb,alpha=alpha,mu=mu,minibatch_size=minibatch_size,num_epochs=num_epochs,
               update_type=update_type,calculate_loss=calculate_loss,half_rate=half_rate)

end

function train(X::Array{Float64,1},Y::Float64,nn::NeuralNetwork,vW::NNWeights=zero(nn.weights),vb::NNWeights=zero(nn.biases);
               alpha::Float64=0.01,mu::Float64=0.9,minibatch_size::Int=100,num_epochs::Int=10,update_type::String="momentum",
               calculate_loss::Bool=true,half_rate::Int=1)
  return train(repmat(X,1,1),repmat([Y],1,1),nn,vW,vb,alpha=alpha,mu=mu,minibatch_size=minibatch_size,num_epochs=num_epochs,
               update_type=update_type,calculate_loss=calculate_loss,half_rate=half_rate)

end
#TODO: train for singletons

function updateWeights!(nn::NeuralNetwork, dW::NNWeights, db::NNWeights, vW::NNWeights, vb::NNWeights,
                       alpha::Float64,mu::Float64,m::Int,update_type::String)

  if nn.reg == "L2" || nn.reg == "Ridge"
    reg = deepcopy(nn.weights)

  elseif nn.reg == "L1" || nn.reg == "Lasso"
    reg = deepcopy(nn.weights)
    for i = 1:L
      reg[i] = sign(reg[i])
      #reg[i][find(reg[i])] = 1
    end
  end
  #might be a neater way to do this
  if lowercase(update_type) == "momentum"
    #velocity update
    vW = mu*vW+(-alpha./m)*((dW) + (nn.c*reg))
    vb = mu*vb + (-alpha./m)*db

    #"position" update
    nn.weights += vW
    nn.biases += vb
  elseif lowercase(update_type) == "nesterov"
    #nesterov momentum
    vW_ = deepcopy(vW)
    vb_ = deepcopy(vb)

    vW = mu*vW + (-alpha./m)*(dW + nn.c*reg)
    vb = mu*vb + (-alpha./m)*db

    nn.weights += (-mu)*vW_+(1+mu)*vW
    nn.biases += (-mu)*vb_ + (1+mu)*vb

  elseif lowercase(update_type) == "sgd"
    #vanilla SGD
    nn.weights += (-alpha./m)*(dW+nn.c*reg)
    nn.biases += (-alpha./m)*db

  elseif lowercase(update_type) == "rmsprop"
    vW = mu*vW + (1-mu)*(dW)^2
    vb = mu*vb + (1-mu)*(db)^2

    nn.weights += (-alpha/m)*dW/sqrt(vW+1e-8)
    nn.biases += (-alpha/m)*db/sqrt(vb+1e-8)

  else
    error(string("No update rule: ",update_type))
  end

end

function feedforward(x::Array{Float64,1},nn::NeuralNetwork)
  L = length(nn.biases)
  a = zero(nn.biases)#Dict{Int,Array{Float64,1}}([i=>zeros(length(nn.biases[i])) for i = 1:L])
  z = zero(nn.biases)#0 #same form as a
  a[0] = deepcopy(x)

  #i think i screwed up something here--last level should just be a linear function approximator i think
  #so for i=L, it would be teh same except a[L][:] = z[L][:]
  #maybe have something that would make it a sigmoid for classification or whatever

  for i = 1:L-1 #L
    z[i][:] = nn.weights[i]*a[i-1]+nn.biases[i]
    a[i][:] = nn.neuron(z[i])
    if lowercase(nn.reg) == "dropout"
      a[i][:] .*= ceil(rand(length(a[i][:]))-(1-nn.c))./nn.c
    end
  end
  z[L][:] = nn.weights[L]*a[L-1]+nn.biases[L]
  if nn.output_neuron == sigm || nn.output_neuron == tanh
    a[L][:] = (nn.output_neuron(z[L]).*(nn.yu-nn.yl)./2)+(nn.yl+nn.yu)./2 #nn.output_neuron(z[L]) #
  else
    a[L][:] = nn.output_neuron(z[L])
  end

  return a[L], a,z
end

function backprop(a::NNWeights,z::NNWeights,y::Array{Float64,1},nn::NeuralNetwork)
  L = length(nn.biases)
  dW = zero(nn.weights)
  db = zero(nn.biases)

  #note: can save time if using sigm by letting the gradient be a.*(1-a) rather than sigm(z).*(1-sigm(z))
  del = zero(nn.biases)#Dict{Int,Array{Float64,2}}([i=>zeros(size(1)) for i = 1:L])
  #del[L] = -(y-a[L]).*nn.output_gradient(z[L])#nn.gradient(z[L])#ones(size(z[L]))#squared error
  #del[L] = -sign(y-a[L]).*nn.output_gradient(z[L]) #L1 error
  del[L] = -nn.loss_gradient(y,a[L]).*nn.output_gradient(z[L])
  for l = L:-1:1

    if l > 1
      del[l-1] = (nn.weights[l]'*del[l]).*nn.gradient(z[l-1])
    end
    dW[l] = del[l]*a[l-1]'
    db[l] = del[l]
  end

  return dW, db

end

function backprop(a::NNWeights,z::NNWeights,y::Float64,nn::NeuralNetwork)
  return backprop(a,z,[y],nn)
end

function predict(nn::NeuralNetwork, x::Float64)
  return predict(nn,[x])
end


function predict(nn::NeuralNetwork, x::Array{Float64,1})
  if nn.scale
    x_ = (x-nn.xu)./nn.xs
  else
    x_ = deepcopy(x)
  end
  y, a, z = feedforward(x_,nn)
  return y#.*((nn.yu-nn.yl)/2)+((nn.yl+nn.yu)/2)
end

function predict(nn::NeuralNetwork, X::Array{Float64,2})
  L = length(nn.biases)
  m = size(X,2)
  Y = zeros(length(nn.biases[L]),m)
  for i = 1:m
    y, a, z = predict(nn,X[:,i])
    Y[:,i] = y
  end
  return Y
end

function squash(nn::NeuralNetwork,W::NNWeights,b::NNWeights)
  L = length(W.x)
  assert(L == length(b.x))
  num_var = sum([length(W.x[i])+length(b.x[i]) for i = 1:L])
  v = zeros(num_var)
  #things
  offset = 0
  for i = L:-1:1
    v[1+offset:length(W.x[i])+offset] = W.x[i][:]
    offset += length(W.x[i])
    v[1+offset:length(b.x[i])+offset] = b.x[i][:]
    offset += length(b.x[i])
  end
  return v
end

function squash(nn::NeuralNetwork)
  return squash(nn,nn.weights,nn.biases)
end

function desquash(nn::NeuralNetwork,v::Array{Float64,1})
  W = zero(nn.weights)
  b = zero(nn.biases)
  L = length(W.x)

  offset = 0
  for i = L:-1:1
    W.x[i] = reshape(v[1+offset:length(nn.weights.x[i])+offset],size(nn.weights.x[i]))
    offset += length(nn.weights.x[i])
    b.x[i] = v[1+offset:length(nn.biases.x[i])+offset]
    offset += length(nn.biases.x[i])
  end

  return W, b
end

function gradient_check(nn::NeuralNetwork,objective::Function,dW::NNWeights, db::NNWeights,index::Array{Int,1}=Int[],eps::Float64=0.0001)
  if length(index) == 0
    ind1 = rand(1:length(dW.x))
    ind2 = rand(1:length(db.x[ind1]))
    r = rand()
    if r > 0.1
      ind3 = rand(1:size(dW.x[ind1],2))
      index = [ind1;ind2;ind3]
    else
      index = [ind1;ind2]
    end
  end
  if length(index) == 3 #look at weights, not biases, alternatively if length(index) == 3
    grad_ = dW.x[index[2]][index[3],index[4]]
    #make a copy of nn, except with the chosen thing to be incremented accordingly (by a plus and a minus)
  elseif length(index) == 2 #biases if length(index) == 2
    grad_ = db.x[index[2]][index[3]]
  else
    error("Incorrect number of arguments for index (req 2 or 3)")
  end

  grad = (objective(nn1)-objective(nn2))./(2*eps)

  return abs(grad-grad_)
end

end #module
