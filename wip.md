
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


-----
ValueError: pyarrow.lib.IpcWriteOptions size changed, may indicate binary incompatibility. Expected 88 from C header, got 72 from PyObject
is solved by deleting pyarrow and installing version 11.0.0
but it causes datasets package to change version
which leads to an error(AttributeError: module 'datasets' has no attribute 'load_dataset')


-----
pip arrow 11.0.0
conda arrow 12.0.1
pip&conda datasets 2.11.0

conda remove pyarrow
same conda and pip

pip remove pyarrow
conda install pyarrow==11.0.0

now pip and conda are version 11.0.0
and dataset is still 2.11.0

now getting error:ModuleNotFoundError: No module named 'aiohttp'
when importing datasets

conda instal aiohttp
conda install pandas

now import works
but using load_dataset gives error: AttributeError: module 'xxhash' has no attribute 'xxh64'

todo: make evaluate work..

works before installing evaluate

if working on this, freeze env and copy it to the side so I can roll back


1) evaluate is not working

2) define what I want to show gal on monday.