from random import randint
from copy import deepcopy


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def random_crop(img, size=128):
    size = int(size/2) #--> 256
    h, w = img.shape[2:] 
    ch, cw = int(h/2), int(w/2)
    ch_r = randint(ch - 190, ch + 190)
    cw_r = randint(cw - int(cw/2), cw + int(cw/2))
    return img[:, :, ch_r-size:ch_r+size, cw_r-size:cw_r+size]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)