## CNN

### LayerNorm:- (done along the layer (dim = -1))
computer mean and variance then
(x - mean)/sqrt(variance + eps)

### BatchNorm:- (done along the minbatch (dim = 0))
computer Mean and Variance then
(x - mean)/sqrt(variance + eps)

### RMSNorm:- 
compute the mean of squares of x then

x * (1/(torch.sqrt(mean_square + eps)))


### Convolution:-

first -> kernel size,stride, padding are needed
for the input, we take a patch/region and multiply it with the kernel + bias (h_out will (H+2*p-kernel_size)//stride + 1)
either use nested loops or im2col (very slow)
therefore used unfold to create a matrix for matrix multiplication.
out -> (out_channels, in_channels, H_out, W_out)

### Pooling:-
    Select a region(based on kernel size) and get the max element of that region

### CNN:-
conv -> relu -> maxpooling .... -> linear(flatten) -> softmax

### grad accumulation:-
simple --> use a grad accumulation step where you only perform lr_scheduler, optimizer step, gradient clipping for each no of grad_accumulation_steps. Simply gradients will be accumulated for every grad_acc_steps before updating model parameters.

this works best for larger effective batch sizes with limited memory.


### Mistake done in the notebook:- 

negative logits in cross_entropy always

