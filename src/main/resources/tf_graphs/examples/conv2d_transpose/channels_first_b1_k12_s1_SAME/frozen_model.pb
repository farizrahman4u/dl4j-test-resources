
�
in_0Const*
dtype0*�
value�B�"�~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?�+-?HM8>v�,?p�e>@�P=��>$T?�q>H�y?hV?(W�>t�>�3?��D? �	?�19?D��> 8�;��=0��=\�W?��??���=��?���>�L?��?��:>�Z$?�j?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
p
conv2d_transpose/kernelConst*
dtype0*A
value8B6" ���>��=?��V? 漼1~>$�8>�!�>R��>
v
conv2d_transpose/kernel/readIdentityconv2d_transpose/kernel*
T0**
_class 
loc:@conv2d_transpose/kernel
J
conv2d_transpose/biasConst*
dtype0*
valueB"        
p
conv2d_transpose/bias/readIdentityconv2d_transpose/bias*
T0*(
_class
loc:@conv2d_transpose/bias
S
conv2d_transpose/ShapeConst*
dtype0*%
valueB"            
R
$conv2d_transpose/strided_slice/stackConst*
dtype0*
valueB: 
T
&conv2d_transpose/strided_slice/stack_1Const*
dtype0*
valueB:
T
&conv2d_transpose/strided_slice/stack_2Const*
dtype0*
valueB:
�
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape$conv2d_transpose/strided_slice/stack&conv2d_transpose/strided_slice/stack_1&conv2d_transpose/strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
T
&conv2d_transpose/strided_slice_1/stackConst*
dtype0*
valueB:
V
(conv2d_transpose/strided_slice_1/stack_1Const*
dtype0*
valueB:
V
(conv2d_transpose/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape&conv2d_transpose/strided_slice_1/stack(conv2d_transpose/strided_slice_1/stack_1(conv2d_transpose/strided_slice_1/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
T
&conv2d_transpose/strided_slice_2/stackConst*
dtype0*
valueB:
V
(conv2d_transpose/strided_slice_2/stack_1Const*
dtype0*
valueB:
V
(conv2d_transpose/strided_slice_2/stack_2Const*
dtype0*
valueB:
�
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape&conv2d_transpose/strided_slice_2/stack(conv2d_transpose/strided_slice_2/stack_1(conv2d_transpose/strided_slice_2/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
@
conv2d_transpose/mul/yConst*
dtype0*
value	B :
^
conv2d_transpose/mulMul conv2d_transpose/strided_slice_1conv2d_transpose/mul/y*
T0
B
conv2d_transpose/mul_1/yConst*
dtype0*
value	B :
b
conv2d_transpose/mul_1Mul conv2d_transpose/strided_slice_2conv2d_transpose/mul_1/y*
T0
B
conv2d_transpose/stack/1Const*
dtype0*
value	B :
�
conv2d_transpose/stackPackconv2d_transpose/strided_sliceconv2d_transpose/stack/1conv2d_transpose/mulconv2d_transpose/mul_1*
N*
T0*

axis 
�
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stackconv2d_transpose/kernel/read	in_0/read*
T0*
data_formatNCHW*
	dilations
*
paddingSAME*
strides
*
use_cudnn_on_gpu(
�
conv2d_transpose/BiasAddBiasAdd!conv2d_transpose/conv2d_transposeconv2d_transpose/bias/read*
T0*
data_formatNCHW 