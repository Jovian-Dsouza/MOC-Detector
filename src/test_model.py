import torch 
from MOC_utils.model import create_model

if __name__ == '__main__':
    num_classes = 24
    K = 7
    arch = 'resnet_18'
    branch_info = {'hm': num_classes,
                        'mov': 2 * K,
                        'wh': 2 * K}
    head_conv = 256

    model = create_model(arch, branch_info, head_conv, K)

    input_shape = (1, 3, 288, 288)
    x = [torch.randn(input_shape)] * K 

    # y = model.backbone(x) #1, 64, 72, 72
    y = model(x)

    # print(len(y))
    print(y[0].keys())
    hm = y[0]['hm']
    mov = y[0]['mov']
    wh = y[0]['wh']

    print(hm.shape)
    print(mov.shape)
    print(wh.shape)