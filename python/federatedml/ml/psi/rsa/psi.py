from fate_arch.tensor import Context
from fate_arch.tensor import GUEST, HOST
from fate_arch.data import DataFrame
from federatedml.secureprotol.encrypt import RSAEncrypt
from federatedml.param import PSIParam
import hashlib


class PSIBase(object):
    def __init__(self, **kwargs):
        self.params = PSIParam.update(kwargs)

    def _post_process(self, data, intersect_index):
        if self.params.only_output_keys:
            return DataFrame(index=intersect_index, schema={"index": data.schema["sid"]})
        else:
            intersect_data = data.join(intersect_index, rebalance=True)
            return intersect_data


class PSIRSAGuest(PSIBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, ctx, data):
        n, e = ctx.pull(HOST, "rsa_public_key")
        index_tensor = ctx.to_tensor(data.index)
        hash_index_tensor = ctx.string_to_arithmetic.apply_(index_tensor, _hash_func, mod=n)
        rand_tensor = ctx.arithmetic_tensor.rand_tensor(index_tensor.shape, mod=n)
        rand_tensor_pow = rand_tensor ** e
        guest_public_sign_tensor = hash_index_tensor * rand_tensor_pow

        ctx.push(GUEST, "guest_public_sign_tensor", guest_public_sign_tensor)
        host_priv_sign_tensor = ctx.push(HOST, "host_priv_sign_tensor")

        guest_priv_sign_tensor = ctx.pull(HOST, "guest_priv_sign_tensor")
        guest_priv_sign_tensor_denoise = guest_priv_sign_tensor / rand_tensor

        double_hash_guest_priv_tensor = ctx.apply_(guest_priv_sign_tensor_denoise, _hash_func)

        rev_df = DataFrame(index=double_hash_guest_priv_tensor, values=index_tensor)
        host_df = DataFrame(index=host_priv_sign_tensor)

        intersect_df = rev_df.join(host_df)
        intersect_index = intersect_df.values
        ctx.push(HOST, "intersect_ids", intersect_index)

        self._post_process(data, intersect_df)


class PSIRSAHost(PSIBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, ctx, data):
        rsa_obj = RSAEncrypt()
        rsa_obj.generate_key(1024)
        n, e, d = rsa_obj.n, rsa_obj.e, rsa_obj.d
        ctx.push(GUEST, "rsa_public_key", (n, e))
        index_tensor = ctx.to_tensor(data.index)
        hash_index_tensor = ctx.string_to_arithmetic.apply_(index_tensor, _hash_func, mod=n)
        host_priv_sign_tensor = ctx.apply_(hash_index_tensor, _hash_func)
        ctx.push(GUEST, "host_priv_sign_tensor", host_priv_sign_tensor)

        guest_public_sign_tensor = ctx.pull(GUEST, "guest_public_sign_tensor")
        guest_priv_sign_tensor = guest_public_sign_tensor ** d
        ctx.push(GUEST, "guest_priv_sign_tensor", guest_priv_sign_tensor)

        intersect_index = ctx.pull(GUEST, "intersect_index")

        self._post_process(data, intersect_index)


def _hash_func(x):
    return int(hashlib.sha256(bytes(x, encoding='utf-8')).hexdigest(), 16)

