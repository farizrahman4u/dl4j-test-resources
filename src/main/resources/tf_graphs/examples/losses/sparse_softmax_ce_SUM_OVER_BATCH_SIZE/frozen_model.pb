
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
Y
in_2Const*
dtype0*=
value4B2
"(ho?�sy>        ��<p-(>b"j?���>숩>    
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
�
3sparse_softmax_cross_entropy_loss/xentropy/xentropy#SparseSoftmaxCrossEntropyWithLogits	in_1/read	in_0/read*
T0*
Tlabels0
X
Psparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_successNoOp
�
%sparse_softmax_cross_entropy_loss/MulMul3sparse_softmax_cross_entropy_loss/xentropy/xentropy	in_2/readQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
T0
�
'sparse_softmax_cross_entropy_loss/ConstConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB: 
�
%sparse_softmax_cross_entropy_loss/SumSum%sparse_softmax_cross_entropy_loss/Mul'sparse_softmax_cross_entropy_loss/Const*
T0*

Tidx0*
	keep_dims( 
�
.sparse_softmax_cross_entropy_loss/num_elementsConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
value	B :

�
3sparse_softmax_cross_entropy_loss/num_elements/CastCast.sparse_softmax_cross_entropy_loss/num_elements*

DstT0*

SrcT0
�
)sparse_softmax_cross_entropy_loss/Const_1ConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB 
�
'sparse_softmax_cross_entropy_loss/Sum_1Sum%sparse_softmax_cross_entropy_loss/Sum)sparse_softmax_cross_entropy_loss/Const_1*
T0*

Tidx0*
	keep_dims( 
�
+sparse_softmax_cross_entropy_loss/Greater/yConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
�
)sparse_softmax_cross_entropy_loss/GreaterGreater3sparse_softmax_cross_entropy_loss/num_elements/Cast+sparse_softmax_cross_entropy_loss/Greater/y*
T0
�
)sparse_softmax_cross_entropy_loss/Equal/yConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
�
'sparse_softmax_cross_entropy_loss/EqualEqual3sparse_softmax_cross_entropy_loss/num_elements/Cast)sparse_softmax_cross_entropy_loss/Equal/y*
T0
�
1sparse_softmax_cross_entropy_loss/ones_like/ShapeConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB 
�
1sparse_softmax_cross_entropy_loss/ones_like/ConstConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *  �?
�
+sparse_softmax_cross_entropy_loss/ones_likeFill1sparse_softmax_cross_entropy_loss/ones_like/Shape1sparse_softmax_cross_entropy_loss/ones_like/Const*
T0*

index_type0
�
(sparse_softmax_cross_entropy_loss/SelectSelect'sparse_softmax_cross_entropy_loss/Equal+sparse_softmax_cross_entropy_loss/ones_like3sparse_softmax_cross_entropy_loss/num_elements/Cast*
T0
�
%sparse_softmax_cross_entropy_loss/divRealDiv'sparse_softmax_cross_entropy_loss/Sum_1(sparse_softmax_cross_entropy_loss/Select*
T0
�
,sparse_softmax_cross_entropy_loss/zeros_likeConstQ^sparse_softmax_cross_entropy_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
�
'sparse_softmax_cross_entropy_loss/valueSelect)sparse_softmax_cross_entropy_loss/Greater%sparse_softmax_cross_entropy_loss/div,sparse_softmax_cross_entropy_loss/zeros_like*
T0 