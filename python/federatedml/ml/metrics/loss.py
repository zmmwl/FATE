class SigmoidBinaryCrossEntropyLoss(object):
    @staticmethod
    def compute_loss(y_true, y_pred, reduction="mean"):
        loss = y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()

        return __reduce(loss, reduction)

    @staticmethod
    def compute_grad(y_true, y_pred, reduction="mean"):
        grad = y_pred - y_true

        return __reduce(grad, reduction)


def __reduce(tensor, reduction):
    if not reduction:
        return tensor
    elif reduction == "mean":
        return tensor.sum() / tensor.shape[0]
    else:
        return tensor.sum()
