
�
in_0Const*
dtype0*�
value�B�"�~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?�+-?HM8>v�,?p�e>@�P=��>$T?�q>H�y?hV?(W�>t�>�3?��D? �	?�19?D��> 8�;��=0��=\�W?��??���=��?���>�L?��?��:>�Z$?�j?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
�
conv2d/kernelConst*
dtype0*�
valuexBv"`ng�>X�>��?���@� >�p�=�f(>~z�>��>��(��>���>LU�>�@�>�n"���q>��P�0]�>v� ?��=�$�>,Y	����þ
X
conv2d/kernel/readIdentityconv2d/kernel*
T0* 
_class
loc:@conv2d/kernel
D
conv2d/biasConst*
dtype0*!
valueB"            
R
conv2d/bias/readIdentityconv2d/bias*
T0*
_class
loc:@conv2d/bias
�
conv2d/Conv2DConv2D	in_0/readconv2d/kernel/read*
T0*
data_formatNHWC*
	dilations
*
paddingSAME*
strides
*
use_cudnn_on_gpu(
Z
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/bias/read*
T0*
data_formatNHWC
*

conv2d/EluEluconv2d/BiasAdd*
T0 