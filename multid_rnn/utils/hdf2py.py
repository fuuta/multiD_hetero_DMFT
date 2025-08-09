import numpy as np
import h5py
import inspect

np_class = tuple(list([c[1] for c in inspect.getmembers(np, inspect.isclass)]))


def get_hdfvalue(f, x, verbose=0):
    # xtype = type(x)
    if isinstance(x, np.void):
        if verbose >= 2:
            print("void")
        return [get_hdfvalue(f, i) for i in x]
    elif isinstance(x, h5py._hl.dataset.Dataset):
        if verbose >= 2:
            print("data")
        return get_hdfvalue(f, x[()])
    elif isinstance(x, h5py.h5r.Reference):
        if verbose >= 2:
            print("ref")
        return get_hdfvalue(f, f[x])
    elif isinstance(x, np.ndarray):
        if x.dtype == object:
            if verbose >= 2:
                print(x.dtype)
            return [get_hdfvalue(f, i) for i in x]
        else:
            if verbose >= 2:
                print(x.dtype)
            return x
    elif isinstance(x, bytes):
        if verbose >= 2:
            print("byte char")
        return x.decode("utf-8")
    elif isinstance(x, np_class):
        if verbose >= 2:
            print("np class {}".format(type(x)))
        return x
    else:
        print("{} type is not hundled".format(type(x)))
        return x


def get_dataset_jl(hdfpath):
    def remove_dup_from_block(list_x):
        for i, x in enumerate(list_x):
            if i != 0:
                # print(x.shape, x[1:].shape)
                list_x[i] = x[1:]
        return list_x

    def concat_blocks(f, name):
        l_data = list(np.asarray(get_hdfvalue(f, f.get(name))))
        # nd_data_raw_shape = np.concatenate(l_data, axis=0).shape
        nd_data = np.concatenate(remove_dup_from_block(l_data), axis=0)
        # print("remove_dup_from_block {} -> {}".format(nd_data_raw_shape, nd_data.shape))
        return nd_data

    ret = {
        "hdfpath": hdfpath,
    }
    with h5py.File(hdfpath, "r") as f:
        data_keys = f.keys()
        paramsA = get_hdfvalue(f, f.get("paramsA"))
        paramsA_dict = dict(map(lambda i, j: (i, j), paramsA[0], paramsA[1]))
        toName = get_hdfvalue(f, f.get("toName"))
        toName_dict = dict(map(lambda i, j: (i, j), toName[0], toName[1]))

        gammas = 1 / paramsA_dict["hetero_tauA"]
        meanGamma = toName_dict["meanGamma"]
        beta = toName_dict["beta"]
        dt = get_hdfvalue(f, f.get("dt"))
        W = get_hdfvalue(f, f.get("W"))

        if f.get("traces") is None:
            # traces = np.asarray(get_hdfvalue(f, f.get("tracesx")))
            # traces = np.concatenate(list(traces), axis=0)
            # ret["tracesx"] = traces
            ret["tracesx"] = concat_blocks(f, "tracesx")

            # tracesa = np.asarray(get_hdfvalue(f, f.get("tracesa")))
            # tracesa = np.concatenate(list(tracesa), axis=0)
            # ret["tracesa"] = tracesa
            ret["tracesa"] = concat_blocks(f, "tracesa")
        else:
            traces = np.asarray(get_hdfvalue(f, f.get("traces")))
            traces = np.concatenate(list(traces), axis=0)
            ret["tracesx"] = traces

        ret["data_keys"] = data_keys
        ret["paramsA_dict"] = paramsA_dict
        ret["toName_dict"] = toName_dict
        ret["gammas"] = gammas
        ret["meanGamma"] = meanGamma
        ret["beta"] = beta
        ret["dt"] = dt
        ret["W"] = W

        if "tracesdx" in data_keys:
            # tracesdx = np.asarray(get_hdfvalue(f, f.get("tracesdx")))
            # tracesdx = np.concatenate(list(tracesdx), axis=0)
            # ret["tracesdx"] = tracesdx
            ret["tracesdx"] = concat_blocks(f, "tracesdx")

        if "tracesda" in data_keys:
            # tracesda = np.asarray(get_hdfvalue(f, f.get("tracesda")))
            # tracesda = np.concatenate(list(tracesda), axis=0)
            # ret["tracesda"] = tracesda
            ret["tracesda"] = concat_blocks(f, "tracesda")

        if "tracesk1x" in data_keys:
            # tracesk1x = np.asarray(get_hdfvalue(f, f.get("tracesk1x")))
            # tracesk1x = np.concatenate(list(tracesk1x), axis=0)
            # ret["tracesk1x"] = tracesk1x
            ret["tracesk1x"] = concat_blocks(f, "tracesk1x")

        if "tracesk1a" in data_keys:
            # tracesk1a = np.asarray(get_hdfvalue(f, f.get("tracesk1a")))
            # tracesk1a = np.concatenate(list(tracesk1a), axis=0)
            # ret["tracesk1a"] = tracesk1a
            ret["tracesk1a"] = concat_blocks(f, "tracesk1a")

        return ret


def get_dataset_py(hdfpath):
    ret = {
        "hdfpath": hdfpath,
    }
    with h5py.File(hdfpath, "r") as f:
        data_keys = f.keys()
        print(data_keys)

        traces = get_hdfvalue(f, f.get("traces"))
        gammas = get_hdfvalue(f, f.get("gammas"))
        J = get_hdfvalue(f, f.get("J"))
        init_state = get_hdfvalue(f, f.get("init_state"))
        params = dict(f.attrs)

        ret["traces"] = traces
        ret["gammas"] = gammas
        ret["J"] = J
        ret["init_state"] = init_state
        return ret, params
