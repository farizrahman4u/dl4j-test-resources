
Y
in_0Const*
dtype0*=
value4B2
"(                              
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
�
in_1Const*
dtype0*�
value�B�
"���徰:�?84�>�S=�5?��7�hF�?�G?Tn�?(��>����n?ʂ@�W}?�?j4�>�-�>��z�C>���W�Y�?�]>�����
�3��=L��?�a)��R��a�����f�4��ӻ�4�?Qڶ?v�5��6�?�-����H>E�X��6�= �K�H�����|���n>%�?ΘĿ��?+F����
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
]
in_2Const*
dtype0*A
value8B6
"(ho?�sy>        ��<p-(>b"j?���>숩>    
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
h
)sparse_softmax_cross_entropy_loss/SqueezeSqueeze	in_2/read*
T0*
squeeze_dims

���������
�
3sparse_softmax_cross_entropy_loss/xentropy/xentropy#SparseSoftmaxCrossEntropyWithLogits	in_1/read	in_0/read*
T0*
Tlabels0
X
Psparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_successNoOp
�
%sparse_softmax_cross_entropy_loss/MulMul3sparse_softmax_cross_entropy_loss/xentropy/xentropy)sparse_softmax_cross_entropy_loss/SqueezeQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
T0
�
'sparse_softmax_cross_entropy_loss/ConstConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB: 
�
%sparse_softmax_cross_entropy_loss/SumSum%sparse_softmax_cross_entropy_loss/Mul'sparse_softmax_cross_entropy_loss/Const*
T0*

Tidx0*
	keep_dims(  