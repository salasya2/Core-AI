### Inplace operations:-

add_
t_
copy_
sub_

### concat operations:

torch.stack() add news new dimension (4,5) --> (4,2,5) depends on where you add it 
torch.cat() adds along existing dimension (4,5) --> (8,5) or (4,10)

###

torch.gather -> indexing 
torch.gather(input, -1, targets) -> it access elements using targets as indices along dimension -1.

###
matmul -> dot product
mul -> cross product

### misc

zeros(shape)
ones(shape)
rand(shape) -> range(-1,1)
randn(shape) -> greater range

zeros_like(tensor)
ones_like(tensor)
rand_like(tensor)
randn_like(tensor)

### Autograd

autograd stores the function in tensor.grad_fn(), we need to initialize the tensors with requires_grad = True to turn them into trainable parameters. Autograd creates a computational graph to execute and calculate the loss. with torch.no_grad() you wont compute the backwardprop.

### optimizer

AdamW optimizer:-

for each state, we have values t, m, v where t is time, m is mean, v is variance.
using m,v values from previous we calculate new m and v;

alpha_t = alpha * math.sqrt((1-beta2\**t)/(1-beta1\**t)) -> bias corrected learning rate, since initially m and v are 0 and this can't be accurate.
p.data -= p.data* weight_decay * alpha -> this tends the weight to be initially towards 0 independently of the gradient (differing step with Adam)

new_m = m(beta1) + (1-beta1)*grad -> 90% of old history + 10% of new history. exponential moving average of gradients.
new_v = v(beta2) + (1-beta2)*math.square(grad) -> exponential moving average of squared gradient. parameters with large grad will create large v -> smaller steps

p.data -= alpha_t*(m/(math.sqrt(v)+eps)) -> large parameter (high v)gradient smaller steps, small gradients(low v) larger steps, effectively gives adaptable steps.

### cosine lr

this makes the learning rate adaptable, at warmup m and v are 0, so jumping to lr of 0.1 will make bad early update harder to recover

cosine decay goes slow at the start, picksup in the middle and slow at the end ensuring model settles gently at the end. linear scheduler decays at constant rate. (it descend from max_learning_rate -> min_learning_rate)

after cosine iterations, it holds lr at min_lr making sure that lr != 0 even if you run extra iters.

warump should 5% of total_iter
cosine should be until 90% of total_iter

### gradient clipping

clips the gradients so that they don't explode

norm = sqrt(sum(grad**2))
clip it to min(1,max_norm/(norm+1e-6))
multiply the grads by clip coeffiecient

### cross entropy
first compute -(log_softmax) - logits 

then loss = mean of alongs logits at the target indices

###  backprop

crossentropy backward = p_i(1-p_i) for target == predicted or p_i if target!=predicted
p_i = e_i/sum(e)

dp_i/dz_i = pi(1-pi)

out = xW_t+b
dout/dw = x
dL/dw = dL/dout * dout/dw => dw = dout*X
dL/dw = dL/dp_i * d_pi/dw
calculate dw which is logit@X.T
db = logit.sum()
dx = logit@W


### training

get raw outputs - > output=model(input)
get loss function -> loss(output,target)
lr_scheduler -
zero out grads ->optimizer.zero_grad()
backward step -> loss.backward()
clipgradients
optimizer step -> optimizer.step()