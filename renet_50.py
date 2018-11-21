# -*- coding: utf-8 -*-
from torch import nn
import torch as t
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    # 实现子module: Residual    Block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            #__init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True)
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    # 实现主module:ResNet34
    # ResNet34包含多个layer,每个layer又包含多个residual block
    # 用子module实现residual block , 用 _make_layer 函数实现layer
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 重复的layer,分别有3,4,6,3个residual block
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # 构建layer,包含多个residual block
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__=="__main__":
    # model = ResNet34()
    # print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    # input = t.autograd.Variable(t.randn(1, 3, 224, 224))
    # o = model(input)
    # exit(0)
    print("-----------------------------")
    import struct
    a = 20
    b = 1
    str = struct.pack("ii", a,b)
    print("str len:", len(str))
    print(type(str), str)
    c, d  = struct.unpack("ii", str) # 返回元组
    print(c, d)
    print("len: ", struct.calcsize('i'))  # len:  4
    print("-----------------------------")
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    a = np.random.randn(4, 4)
    a = a.astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)

    mod = SourceModule("""
    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *=2; 
    }
    """)
    func = mod.get_function("doublify")
    func(a_gpu, block=(4,4,1))
    a_double = np.empty_like(a)
    cuda.memcpy_dtoh(a_double, a_gpu)
    print(a_double)
    print(a)

    import pycuda.gpuarray as gpuarray
    a_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))
    a_double = (a_gpu*2).get()
    print(a_double)
    print(a)

    print("---------------------------")
    mod = SourceModule("""
    struct DoubleOperation {
        int datalen, __padding; // so 64-bit ptrs can be aligned
        float *ptr;
    };

    __global__ void double_array(DoubleOperation *a) {
        a = &a[blockIdx.x];
        for (int idx = threadIdx.x; idx < a->datalen; idx += blockDim.x) {
            a->ptr[idx] *= 2;
        }
    }
    """)


    class DoubleOpStruct:
        mem_size = 8 + np.intp(0).nbytes

        def __init__(self, array, struct_arr_ptr):
            self.data = cuda.to_device(array)
            self.shape, self.dtype = array.shape, array.dtype
            cuda.memcpy_htod(int(struct_arr_ptr), np.getbuffer(np.int32(array.size)))
            cuda.memcpy_htod(int(struct_arr_ptr) + 8, np.getbuffer(np.intp(int(self.data))))

        def __str__(self):
            return str(cuda.from_device(self.data, self.shape, self.dtype))


    struct_arr = cuda.mem_alloc(2 * DoubleOpStruct.mem_size)
    temp1 = int(struct_arr)
    temp2= DoubleOpStruct.mem_size
    temp3 = np.int32(34)
    print(temp3.nbytes)
    temp4 = np.getbuffer(temp3)
    do2_ptr = int(struct_arr) + DoubleOpStruct.mem_size

    array1 = DoubleOpStruct(np.array([1, 2, 3], dtype=np.float32), struct_arr)
    array2 = DoubleOpStruct(np.array([0, 4], dtype=np.float32), do2_ptr)
    print("original arrays", array1, array2)

    func = mod.get_function("double_array")
    func(struct_arr, block=(32, 1, 1), grid=(2, 1))
    print("doubled arrays", array1, array2)

    func(np.intp(do2_ptr), block=(32, 1, 1), grid=(1, 1))
    print("doubled second only", array1, array2, "\n")