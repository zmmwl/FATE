from fate_arch import Context, Guest, Host
from fate_arch import DataLoader
from fate.param import PsiParam
import fate_arch
import pandas as pd, path


class PSI(object):
    def __init__(self, context, **kwargs):
        self._psi_param = PsiParam(**kwargs)
        self._ctx = context
        if self._ctx.role == Guest:
            self._runner = PSIGuest(self._psi_param)
        else:
            self._runner = PSIHost(self._psi_param)

    def fit(self, data):
        return self._runner.fit(data)


class PSIGuest(PSI):
    def fit(self, data):
        df = DataLoader.load_data(data)
        ...


class PSIHost(PSI):
    ...


# Guest Code:
from fate.apps import PSI
from fate.apps import FeatureBinning
from fate.apps import SSHELR


ctx = fate_arch.Context.create_context()
df = pd.load_from_csv(path)

psi_runner = PSI(ctx, **psi_param)
ret_psi = psi_runner.fit(df)

binning_runner = FeatureBinning(ctx, **binning_param)
ret_binning = binning_runner.fit(ret_psi)

sshe_lr_runner = SSHELR(ctx, **lr_param)
predict_result = sshe_lr_runner.fit(ret_binning)

