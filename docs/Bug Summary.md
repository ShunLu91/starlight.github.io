---
layout: default
title: Bug Summary
nav_order: 2
---


PSPNet (Semantic Segmentation)
剪枝问题记录
1. 下采样层，由于residule带有卷积，residual剪枝后和主分支剪枝后的通道相加不对应
  1. 解决：设定residule卷积输出通道与该block最后一个卷积输出通道的被剪枝index相同
2. 微调时，模型测试时间太长
  1. 解决：仅使用10张图片进行模型测试，判断停止点

DeepLabV3 (Semantic Segmentation)
剪枝问题记录
1. 网络最后的输出层被剪枝，导致语义分割任务的输出类别不一致
  1. 原因：使用自动导出的方法容易存在该问题，无法自动处理
  2. 解决：使用手动剪枝
2. 残差块的第一个带下采样的层，通道无法对齐
  1. 原因：由于in_place_dict中结构顺序撰写错误导致
  2. 解决：修正每个残差块的第一个下采样层的输入为残差支路的x而不是带卷积支路的x
3. ASPP通道无法对齐
  1. 原因：金字塔结构是并联的5个分支，每个分支的输入均为金字塔整体的外部输入，由于in_place_dict中将金字塔内部的连接关系定义错误，因此导致报错
  2. 解决：修正5个分支的输入均为同一个外部输入x

CFNet (Depth Prediction)
剪枝问题记录
1. 环境安装：ModuleNotFoundError: No module named 'OpenEXR'
  1. 解决：建议使用如下步骤安装该软件包。
apt-get update
apt-get install libopenexr-dev
export CFLAGS="-I/Users/USERNAME/homebrew/include/OpenEXR -std=c++11"
export LDFLAGS="-L/Users/USERNAME/homebrew/lib"
pip install OpenEXR
  2. 可能出现的问题1：OpenEXR.cpp:36:10: fatal error: 'ImathBox.h' file not found
    1. 解决：apt-get install libopenexr-dev
    2. 引用：https://github.com/AcademySoftwareFoundation/openexr/issues/449
  3. 可能出现的问题2：fatal error: 'ImathBox.h' file not found
    1. 解决：需要按照如下方式设置编译用的std
    2. 引用：https://github.com/google-research/kubric/issues/19
export CFLAGS="-I/Users/USERNAME/homebrew/include/OpenEXR -std=c++11"
export LDFLAGS="-L/Users/USERNAME/homebrew/lib"
pip install OpenEXR
2. 两个输入问题
  1. 问题：网络包含两个输入
  2. 解决：将剪枝器（Pruner）如FPGMPruner的输入参数dummy_input改为元组(rand_inputs1, rand_inputs2)
3. 检查view维度不一致报错
  1. 问题：一个节点的输入只有一个维度
  2. 解决：加入维度数量判断
[图片]
4. Update_mask()函数维度不一致报错
[图片]
  1. 解决：nni在构图时，其中两个卷积的前驱与后继判断错误。提前获取这两个卷积在网络中的名称，剪枝时设定剪枝算法跳过这两个卷积，不对它们进行剪枝即可。
5. 模型尺寸过大，剪枝时单张卡放不下
  1. 解决：Debug剪枝时，先使用特别小的输入尺寸；确认剪枝可正常进行后，设定BatchSize为最小值，进行剪枝。
6. 模型剪枝完成，但后续的3D卷积没有剪枝，导致模型剪枝部分的最后一层输出与后续未剪枝部分的输入不匹配
[图片]
  1. 解决：剪枝部分的最后一个卷积的输出通道全都不剪
量化问题记录
1. 节点无法识别，而且此时生成的ONNX模型无法用Netron打开
[图片]
  1. 解决记录：查到这个问题是由于前向传播中用到Tensor切片导致的。因此有两种解决方案：解法1：去掉前向传播中的切片操作；解法2：加入切片操作的插件
  2. 解法1测试：
    1. 由于在前向传播中，这个模型对相同的模块进行了两次重复调用，所以在剪枝时将两次调用合并，对结果进行切片。由于该切片操作在量化时出现问题，因此这里还是将前向传播调整回原始模式进行测试，即两次重复调用，不进行切片。
    2. 结果：发现会出现同样的问题，看来这个问题无法绕开。因为在网络中的其它函数里，出现了大量的切片操作。
[图片]
[图片]
[图片]
  3. 解法2测试：
    1. 添加可识别上述操作的插件生成，细节请参照：课题二检测Yolo5
  4. ONNX无法打开：问题在于网络结构太大，过于复杂。因此，可以将网络中出问题的那部分单独提取为一个新的网络，然后单独到ONNX就可以打开了。
2. 3D反卷积量化报错
[图片]
  1. 问题：根据错误提示，发现TensorRT还不支持不对称的反卷积。官方回复将在未来支持。
[图片]
  2. 解决：上述问题主要由于3d deconv中的output_padding=1导致，因此将其设为0后，手动对输出结果进行补0即可，前向传播代码修改如下：
# 报错代码
# conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
# conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

# 正确代码
y = self.conv5(conv4)
y_pad = torch.zeros_like(y)
y = torch.cat((y, y_pad[:, :, 0, :, :].unsqueeze(dim=2)), dim=2)
y_pad = torch.zeros_like(y)
y = torch.cat((y, y_pad[:, :, :, 0, :].unsqueeze(dim=3)), dim=3)
y_pad = torch.zeros_like(y)
y = torch.cat((y, y_pad[:, :, :, :, 0].unsqueeze(dim=4)), dim=4)
y = self.conv5bn(y)
conv5 = FMish(y + self.redir2(conv2))

y = self.conv6(conv5)
y_pad = torch.zeros_like(y)
y = torch.cat((y, y_pad[:, :, 0, :, :].unsqueeze(dim=2)), dim=2)
y_pad = torch.zeros_like(y)
y = torch.cat((y, y_pad[:, :, :, 0, :].unsqueeze(dim=3)), dim=3)
y_pad = torch.zeros_like(y)
y = torch.cat((y, y_pad[:, :, :, :, 0].unsqueeze(dim=4)), dim=4)
y = self.conv6bn(y)
conv6 = FMish(y + self.redir1(x))
3. 量化导出时处理ONNX中的IF结构报错
[图片]
  1. 问题：将出错的一段模型结构单独量化后进行可视化如下，发现该问题是由于模型中存在torch.squeeze(x,dim=1)造成的。量化时，需要判断第1维是否为0，如果为0就去掉这一维。
[图片]
  2. 解决：前向传播时可以明确第一维为0因此无需判断，直接使用正常索引即可，代码修改如下
pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
# pred2_s4 = torch.squeeze(pred2_s4, 1) # 报错代码
pred2_s4 = pred2_s4[:, 0, :, :] # 正确代码

pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                         align_corners=True)
# pred1_s3_up = torch.squeeze(pred1_s3_up, 1) 
pred1_s3_up = pred1_s3_up[:, 0, :, :]

pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
# pred1_s2 = torch.squeeze(pred1_s2, 1)
pred1_s2 = pred1_s2[:, 0, :, :]
4. 量化导出时处理ONNX中的Clip结构报错
  1. 问题：Assertion failed: inputs.at(2).is_weights() && "Clip max value must be an initializer!"
  2. 解决：经过查询，该问题是由于torch.clamp函数中未设置max参数导致，在所有torch.clamp的位置添加一个max参数即可。
5. 量化导出时处理ONNX中的torch.gather结构报错
  1. 使用量化代码导出时出现如下错误：
[图片]
  2. 上述错误未包含明确信息，因此使用trt_infer.py代码进行调试，出现如下报错：
[图片]
  3. 根据上述错误可知，是缺少处理'GatherElements'的支持。因为torch.gather函数在导出为ONNX后，会形成GatherElements节点，而我们使用的7.1.3.4版本TensorRT不支持处理该节点。
[图片]
  4. 经过查询发现：TensorRT8.0支持GatherElements操作，以内置函数的方式提供，而不是插件，内置函数是闭源的，所以没有能直接参考的。
  5. 解决：为了绕开上述问题，将模型分成两段。前半段进行量化，后半段直接使用原始模型。测试时，将第一段的输出转为Tensor后直接送入第二段即可。或者，寻找一种可量化的操作替换该不可量化操作。
[图片]
EfficientNet-B3 (Semantic Segmentation)
剪枝问题记录
1. Tensor和int数值无法直接相除
[图片]
  1. 解决：该问题可能源于不同Pytorch版本对除法的处理不一致。因此，将所有Tensor转为数值即可。
input_size = []
for _size in input_size_ori:
    if torch.is_tensor(_size):
        input_size.append(_size.item())
    else:
        input_size.append(_size)
2. 使用NNI自动剪枝构图报错
[图片]
  1. 解决：调试发现原因在于无法识别自定义的模块MemoryEfficientSwish，该模块是一个激活函数，可以直接跳过。因此，改写tensorboard文件夹下的_pytorch_graph.py文件设置跳过该函数即可。
class NodePyOP(NodePy):
    def __init__(self, node_cpp):
        super(NodePyOP, self).__init__(node_cpp, methods_OP)
        # Replace single quote which causes strange behavior in TensorBoard
        # TODO: See if we can remove this in the future
        try:
            self.attributes = str({k: node_cpp[k] for k in node_cpp.attributeNames()}).replace("'", ' ')
            self.kind = node_cpp.kind()
        except:
            # raise ValueError('error')
            self.attributes = []
            self.kind = node_cpp.kind()
3. 代码卡在_get_parent_layers()中死循环
  1. 问题：lib/compression/pytorch/utils/shape_dependency.py中ChannelDependency()的_get_parent_layers()函数
  2. 解决：避免遍历已遍历过的节点即可。添加一个travel_list，对于已经遍历过的节点不再遍历。
[图片]
4. 无法识别自定义的Conv2dStaticSamePadding()模块
[图片]
  1. 问题：无法识别自定义的卷积层Conv2dStaticSamePadding()，导致后续层没有输入。
  2. 解决：测试如下两种解决方案，测试后采用第二种：
    1. 在lib/compression/pytorch/speedup/jit_translate.py中添加无法识别的操作函数。但对于卷积函数F.conv2d()，需要输入很多形参，而在识别时形参并没有传过来。因此，不推荐该方案；
    2. 将无法识别的Conv2dStaticSamePadding()模块改成可以识别的情况。即原始的Conv2dStaticSamePadding()是继承了nn.Conv2d，将其改为nn.Module即可，然后检查定义中需要修改的地方。推荐使用此方案。
  3. 使用方案2修改后的代码对比如下所示：
    1. 原始代码，剪枝时无法识别以下模块：
class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """
    # With the same calculation as Conv2dDynamicSamePadding
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w - pad_w // 2, pad_w - pad_w // 2,
                                                pad_h - pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()
    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
    2.  修改后的代码，剪枝时可自动识别：

class Conv2dStaticSamePadding(nn.Module):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """
    # With the same calculation as Conv2dDynamicSamePadding
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.stride = self.conv.stride if len(self.conv.stride) == 2 else [self.conv.stride[0]] * 2
        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.conv.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.conv.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.conv.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w - pad_w // 2, pad_w - pad_w // 2,
                                                pad_h - pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()
    def forward(self, x):
        x = self.static_padding(x)
        x = self.conv(x)
        return x
5. 显存不足报错
  1. 问题：ModelSpeedup()中confidence默认值为8，相当于batch为8，自动导出模型时会报显存不足。但由于已经使用了显存为32G的V100，所以只能把confidence降到2就没问题了。
[图片]
量化问题记录
1. F.interpolate导出为ONNX后，解析报错
  1. 使用opset_version=11，以下为各种不同的定义方式下的报错信息。
  2. x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)), int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=True)
[图片]
  3. x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)), int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=False)
[图片]
  4. x = F.interpolate(x, scale_factor=8.0, mode='bilinear', align_corners=False)
[图片]
  5. x = F.interpolate(x, scale_factor=8.0, mode='bilinear', align_corners=True)
[图片]
  6. x = F.interpolate(x, scale_factor=(8.0, 8.0), mode='bilinear', align_corners=False)
[图片]
2. pycuda._driver.LogicError: cuMemcpyHtoDAsync failed: invalid argument
  1. 问题：该错误由于初始化ModelSpeedupTensorRT设置的input_shape和推理时不一致所导致。
engine = ModelSpeedupTensorRT(
    model,
    input_shape,
    config=None,
    calib_data_loader=calib_loader,
    batchsize=args.batch_size,
    ONNX_path=ONNX_path,
    calibration_cache=cache_path,
    extra_layer_bit=extra_layer_bit,
)
  2. 解决：保持训练和推理数据尺寸一致
3. 检查量化前后模型的输出——左图量化前，右图量化后，存在较大差异
[图片]
  1. 上述问题的根源在于，EfficientNet在解耦字符串时，会将其转换为list，去掉方括号即可解决上述问题。
[图片]
HSMNet (Location)
剪枝问题记录
1. 剪枝后，模型前向传播通道无法对齐。
  1. 解决：根据报错信息，检查通道无法对齐的卷积层的取整问题。
量化问题记录
1. torch.squeeze引发的IF判断无法识别
[图片]
  1. 原因：torch.squeeze压缩维度时，需要判断该维度是否为1。若判断成功则会在ONNX模型中引入IF节点，导致TensorRT转换不成功。
  2. 解决：如下代码所示，不使用squeeze函数，直接使用切片函数即可。
# 下方代码会引发上述错误
# return fvl, costl.squeeze(1)
# 修改为以下代码即可
return fvl, costl[:, 0, :, :, :]
# 和上述相同问题，解决方案也相同
# return pred3, torch.squeeze(entropy)
return pred3, entropy[0, :, :]
2. 加载预训练权重无法获得预训练性能
[图片]
  1. 原因：原始模型是在多卡训练的，预训练权重中的操作名称包含'module.'，导致大部分操作没有加载；而由于该网络预训练权重中的操作和模型定义时的操作略有不同，因此加载权重时设置匹配strict=False，所以没有检查出上述问题，最终导致测试时精度很低。
  2. 解决：去除预训练权重中操作名称的'module.'前缀。
3. 测试集模型输入尺寸全都不一致
[图片]
  1. 解决：此问题说明该定位模型需要处理输入尺寸不一致的数据，而模型量化需要固定模型的输入尺寸。因此，该模型无法进行量化。