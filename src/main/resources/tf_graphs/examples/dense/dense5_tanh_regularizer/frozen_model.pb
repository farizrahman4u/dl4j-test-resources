
�
in_0Const*
dtype0*i
value`B^"P~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
�
dense/kernelConst*
dtype0*i
value`B^"P�0�>��2?*�J?��ؼ �o>(�->�	{>t[�>���>8�/�N�;?.�?T��>��$?0$r�W�>訛�Tt�>��??`�D=
U
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel
K

dense/biasConst*
dtype0*)
value B"                    
O
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias
c
dense/MatMulMatMul	in_0/readdense/kernel/read*
T0*
transpose_a( *
transpose_b( 
W
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC
*

dense/TanhTanhdense/BiasAdd*
T0 