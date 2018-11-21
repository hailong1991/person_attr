
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'checkpoint management'

import os
import torch

def save_checkpoint(state, address, index):
    name = 'model_parameters.pth_%d.tar' % index

    folder = os.path.exists(address)
    if not folder:
        os.mkdir(address)
        print('--- create a new folder ---')
    fulladress = address + '\\' + name
    torch.save(state, fulladress)
    print('model saved:', fulladress)

def load_checkpoint(address):
    name = 'model_parameters.pth'
    fulladress = address + '\\' + name
    return torch.load(fulladress)