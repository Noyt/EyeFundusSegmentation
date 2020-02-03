import numpy as np
import cv2


def ROI(img, threshold=5):
    """
    Assume img to be a uint8, h,w c format image
    :param img: Array
    :return:
    """
    mask = cv2.medianBlur(img[:, :, 2], 21) > threshold

    return mask


class FundusProcessor:
    def __init__(self, img):

        self.img = self.preprocess(img)

        self._OD_mask = None
        self._vessel_mask = None
        self._ROI_mask = None
        self._macula_mask = None

    def preprocess(self, img):
        assert img.ndim == 3
        if img.max() > 1:
            img = img.astype(np.float32) / 255

        if img.shape[0] != 3:
            img = img.transpose((2, 0, 1))

        return img

    @property
    def OD(self):
        if self._OD_mask is None:
            pass

        else:
            return self._OD_mask

    @property
    def vessels(self):
        if self._vessel_mask is None:
            pass
        else:
            return self._vessel_mask

    @property
    def ROI(self):
        if self._ROI_mask is None:
            pass
        else:
            return self._ROI_mask

    @property
    def macula(self):
        if self._macula_mask is None:
            pass
        else:
            return self._macula_mask


L_MEAN = 31.319101
A_MEAN = 17.877468
B_MEAN = 29.181826
L_STD = 12.684237
A_STD = 7.002096
B_STD = 10.272004
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def LAB_clahe(img):
    mask = ROI(img)
    mean_b = np.median(img[:, :, 0][mask])
    mean_g = np.median(img[:, :, 1][mask])
    mean_r = np.median(img[:, :, 2][mask])
    mean_channels = [mean_b, mean_g, mean_r]
    img = np.clip(
        img.astype(np.float32) - cv2.medianBlur(img, 51) * np.expand_dims(mask, 2) + np.asarray(mean_channels).astype(
            np.uint8), 0, 255)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)

    lab = (lab * np.expand_dims(mask, 2)).astype(np.float32)
    lab[:, :, 0] -= lab[:, :, 0].mean()
    lab[:, :, 0] *= L_STD / (lab[:, :, 0].std() + 1e-7)
    lab[:, :, 0] += L_MEAN

    lab[:, :, 1] -= lab[:, :, 1].mean()
    lab[:, :, 1] *= A_STD / (lab[:, :, 1].std() + 1e-7)
    lab[:, :, 1] += A_MEAN

    lab[:, :, 2] -= lab[:, :, 2].mean()
    lab[:, :, 2] *= B_STD / (lab[:, :, 2].std() + 1e-7)
    lab[:, :, 2] += B_MEAN
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return rgb * np.expand_dims(mask, 2)


def pos_var(x):
    return np.mean(abs(x[x >= np.mean(x)] - np.mean(x)) ** 2)


def switch_side(x):
    c, h, w = x.shape
    if pos_var(x[:3, h // 4:3 * h // 4, 20:w // 2 + 20]) > pos_var(x[:3, h // 4:3 * h // 4, w // 2 - 20:-20]):
        return x
    else:
        return x[:, :, ::-1]
