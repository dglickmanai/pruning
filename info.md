outgoing_edges_norm = subset[name].weight.data.norm(p=1,dim=0)
average_logtics = torch.sqrt(wrapped_layers[name].scaler_row) #not necesserly need sqrt

scores = average_logtics * outgoing_edges_norm # this should be after the relu

---
how are the activations stats collected?
wrapped_layers[name].add_batch(inp[0].data, out.data) <-- shape is (in_features, batch_size*seq_len)
reduced to (in_features,1), norm over dim 1.. i.e stats for each activation feature

now element-wise multiple this with the weights(of the linear layer).. so that each weight is "scored" by multipying it with the activatoin stats


-----------------
shapes:
linear: (in_features, out_features) : linear.data.weight: out_features x in_features


both llama models are sharding..


##ideas for wrapping objects

wrapper is an object.
I can pass to it reference of other wrappers/ masks

I can take the first wrapper.. if wrap_layers if name is none 
pass child masks to it some how

** instead of wrapping, we can just compute the thres value such that %s of the total mask weights will be < thres.. then just pass the thres refernce to each mask module
**
