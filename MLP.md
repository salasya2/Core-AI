## MLP on MNIST

### Custom Dataset Class
- needs datafile
- have to define __len__ and __getitem__(for both labels and samples) to create a class.

### Custom DataLoader class
- can do without using Dataloader class inheritance
- have to define __iter__ -> must shuffle the data if shuffle is set to true and then use yield to yield each batch in batches
- collate_fn -> stack input tensors and convert labels to tensor. (have to use zip(*batch) to get raw input and labels.) 
- if you don't use inheritance define len method to access it later

### AdamW
- same as previous --> make sure t  starts at  1.

### MLP
- softmax must be either used in loss fn or in model definition
- if using images first flatten the inputs in the forward method.
- cosine_lr_scheduler when batches -> use a global_step variable inorder to track iters since there might be 1800 samples per batch and 10 epochs making it 18000 steps, so don't initialize warmup ..etc on epoch
- for loss, average it along the batch length.

### helper fns
- for sigmoid using implementing it naively can cause exploding or vanishing gradients since e ^ -x can create large negative numbers for unnormalized inputs, so use torch.sigmoid.
- for softmax, makesure you select [0] for x_max
- for backpropagation, while calculating the dz2 , make sure you convert actual y to one-hot encoding since we will get a softmax distribution for outputs.
- for dw's make sure you devide by len of the batches since, Otherwise  gradients will scale with batch size → unstable training.
- for relu, derivative will be either 0 or 1 depending on z> 1.
- while clipping gradients, make sure you modify the gradients inplace(p.grad.data)
- len of dataset in dataloader is len(dataloader.dataset) 



