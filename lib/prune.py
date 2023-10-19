import torch
import torch.nn as nn
from tqdm import tqdm
import time
from .eval import eval_ppl_wikitext
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT, Wrapper
from .data import get_loaders


def find_layers(module, layers=[nn.Linear], name='', names=[]):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    assert type(
        module) is not Wrapper, 'should not call this w ith wrapper for now.. later should probably just add wrapper to layers..'
    if type(module) in layers and (not names or name.endswith(tuple(names))):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
            , names=names))
    return res


def wrap_model(model, args, names):
    layers = model.model.layers
    return [(layer, wrap_layers(layer, args, names=names)) for layer in layers]


def wrap_layers(module, args, layers=[nn.Linear], name='', names=[]):
    """
    Recursively wrap layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to wrap.
        name (str): Name of the module.

    Returns:
        None. The module is modified in-place.
    """
    if isinstance(names, str):
        names = [names]
    ret = {}
    for name1, child in module.named_children():
        child_name = name + '.' + name1 if name != '' else name1
        if type(child) in layers and (not names or name1.endswith(tuple(names))):
            wrapper = Wrapper(child, args, layer_name=name1, track=False)
            setattr(module, name1, wrapper)
            ret[child_name] = wrapper
        else:
            wrapped_names = wrap_layers(child, args, layers=layers, name=child_name, names=names)
            ret.update(wrapped_names)
    return ret


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if hasattr(model, 'hf_device_map') and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    # forwards the model to get actual activations
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, names=args.weights_to_prune)

        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name], activation_strength_metric=args.activation_strength_metric)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (
                            alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_activations(args, model, tokenizer, dataloader, device=torch.device("cuda:0")):
    wrapped_layers = wrap_model(model, args, names=args.weights_to_prune)
    if args.ignore_init_masking_by_activations:
        return
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # pruning stats collection
    #############
    with torch.no_grad():
        # returns input and output for the first layer
        # inps is (len(dataset), seqlen,input_size).
        # attention_mask is (seqlen,seqlen).. just "diagonal" causal attention mask(same for all inputs).
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # passes layer by layer
    for i in range(len(wrapped_layers)):
        layer, subset = wrapped_layers[i]
        # hook model
        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        for x in subset.values():
            x.track = True

        for j in range(args.nsamples):
            # forward pass to register activations
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for x in subset.values():
            x.track = False

        for name in subset:
            print(f"pruning layer {i} name {name}")
            # prepare prune metric stats
            subset[name].set_prune_stats(args)

        for j in range(args.nsamples):
            with torch.no_grad():
                # forward again to get next layer inputs
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
    ############## pruning stats ends

    model.config.use_cache = use_cache


def train_mask(args, train_loader, testloader, device, model, classification=False):
    for param in model.parameters():
        param.requires_grad = False
    wrapper_layers = [(module_name, module) for lay in
                      [*model.model.layers] for (module_name, module) in lay.named_modules() if
                      type(module) == Wrapper]
    params_to_train = []
    for (module_name, module) in wrapper_layers:
        module.mask.requires_grad = True
        params_to_train.append(module.mask)
    # optimizer = torch.optim.Adam(params_to_train, lr=args.mask_train_lr)
    optimizer = torch.optim.SGD(params_to_train, lr=args.mask_train_lr)

    bs = args.mask_train_bs
    sparsity = args.sparsity_ratio

    for i in tqdm(range(args.mask_train_epochs)):

        if args.gradual_pruning:
            # depenads if running wandb exp or not
            if hasattr(args, 'sparsity_ratio'):
                args.sparsity_ratio = (sparsity * (i + 1)) / args.mask_train_epochs
            else:
                args._items['sparsity_ratio'] = (sparsity * (i + 1)) / args.mask_train_epochs

        # Loop through each batch
        losses = []
        for batch in tqdm(train_loader):
            if not classification:
                loss = language_modeling_train_step(batch, model)
            else:
                print(batch)
                out = model(batch['input_ids'], batch['attention_mask'])
                print(out)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            # print(f"train loss {loss.item()}")

        avg_loss = sum(losses) / len(losses)
        print(f"train loss {avg_loss}")

        stats = {'train_loss': avg_loss}
        if i % 5 == 0:
            with torch.no_grad():
                start = time.time()
                ppl = eval_ppl_wikitext(model, testloader, 8, device)
                print(f"wiki ppl {ppl}. took {time.time() - start} seconds")
                stats['wiki_ppl'] = ppl

        if args.gradual_pruning:
            stats['sparsity_ratio'] = args.sparsity_ratio

        if args.wandb_exp_name is not None and args.wandb_exp_name != "":
            import wandb
            wandb.log(stats)


def language_modeling_train_step(batch, model):
    batch = batch.squeeze(1).to(model.device)
    lm_logits = model(batch).logits
    # Shift logits and labels for next token prediction
    shift_logits = lm_logits[:, :-1, :]
    shift_labels = batch[:, 1:]
    # Compute loss
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    return loss


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        subset = find_layers(layer, names=args.weights_to_prune)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, names=args.weights_to_prune)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                W_mask = (W_metric <= thresh)

            W[W_mask] = 0


def check_sparsity(model, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, names=args.weights_to_prune)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params
