
### loading AutoModelForSequenceClassification
we get an error when trying to load the model.
this happens because the pretrained weights are loaded to gpu devices,
but, as far as I understand, the "new" weights, that are created with the classification head,
are created on the meta device.

when using low_cpu_mem_usage or device_map = 'auto',
init_empty_weights() is called, which creates the new weights on the meta device.

consider try to load in cpu then move to gpu.
the saved model(in .cache/....) is with format of LMHEAD not classification
**load checkpoint and dispatch will not work I think, becuase the model setup is different**
I can try saving the model after I load it with autoseqclassificaiton


18.10
waiting for fourms to reply.. I might have to just do it manually..
anyway, I can start with just working on cpu.......


no truncate and padding...
it does not pad because max_len is not set in llama 2
I can solve this by padding online, with dataloader...
with nous-llama its fine
--
actually problem is that not all lengths is same
len(train_loader.dataset[-1]['input_ids']) != len(train_loader.dataset[0]['input_ids'])

i need to do the padding with the data collator..
but rpmc uses 2 sentences 

ValueError: Trying to set a tensor of shape torch.Size([4, 4096]) in "weight" (which has shape torch.Size([3, 4096])), this look incorrect.
Looks like I saved a model with 4 classes and trying to load it with 3 classes
an option to fix this is to save the model for every individual number of class every time,
i.e check if exists, if not load on cpu and save..