
e
in_0Const*
dtype0*I
value@B>"0~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
e
in_1Const*
dtype0*I
value@B>"0�E?��m?�|?ز�>��$?@�?�n&?��B?ܰB?��>ps?�*\?
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
E
in_2Const*
dtype0*)
value B"��q?�~?�]?�O�>
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
=
absolute_difference/SubSub	in_1/read	in_0/read*
T0
@
absolute_difference/AbsAbsabsolute_difference/Sub*
T0
J
Babsolute_difference/assert_broadcastable/static_dims_check_successNoOp
�
absolute_difference/MulMulabsolute_difference/Abs	in_2/readC^absolute_difference/assert_broadcastable/static_dims_check_success*
T0
�
absolute_difference/ConstConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB"       
x
absolute_difference/SumSumabsolute_difference/Mulabsolute_difference/Const*
T0*

Tidx0*
	keep_dims( 
�
'absolute_difference/num_present/Equal/yConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
k
%absolute_difference/num_present/EqualEqual	in_2/read'absolute_difference/num_present/Equal/y*
T0
�
*absolute_difference/num_present/zeros_likeConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB*    
�
/absolute_difference/num_present/ones_like/ShapeConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB"      
�
/absolute_difference/num_present/ones_like/ConstConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *  �?
�
)absolute_difference/num_present/ones_likeFill/absolute_difference/num_present/ones_like/Shape/absolute_difference/num_present/ones_like/Const*
T0*

index_type0
�
&absolute_difference/num_present/SelectSelect%absolute_difference/num_present/Equal*absolute_difference/num_present/zeros_like)absolute_difference/num_present/ones_like*
T0
�
`absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_dims_check_successNoOpC^absolute_difference/assert_broadcastable/static_dims_check_success
�
Aabsolute_difference/num_present/broadcast_weights/ones_like/ShapeConstC^absolute_difference/assert_broadcastable/static_dims_check_successa^absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_dims_check_success*
dtype0*
valueB"      
�
Aabsolute_difference/num_present/broadcast_weights/ones_like/ConstConstC^absolute_difference/assert_broadcastable/static_dims_check_successa^absolute_difference/num_present/broadcast_weights/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *  �?
�
;absolute_difference/num_present/broadcast_weights/ones_likeFillAabsolute_difference/num_present/broadcast_weights/ones_like/ShapeAabsolute_difference/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0
�
1absolute_difference/num_present/broadcast_weightsMul&absolute_difference/num_present/Select;absolute_difference/num_present/broadcast_weights/ones_like*
T0
�
%absolute_difference/num_present/ConstConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB"       
�
absolute_difference/num_presentSum1absolute_difference/num_present/broadcast_weights%absolute_difference/num_present/Const*
T0*

Tidx0*
	keep_dims( 
�
absolute_difference/Const_1ConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB 
|
absolute_difference/Sum_1Sumabsolute_difference/Sumabsolute_difference/Const_1*
T0*

Tidx0*
	keep_dims( 
�
absolute_difference/Greater/yConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
o
absolute_difference/GreaterGreaterabsolute_difference/num_presentabsolute_difference/Greater/y*
T0
�
absolute_difference/Equal/yConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
i
absolute_difference/EqualEqualabsolute_difference/num_presentabsolute_difference/Equal/y*
T0
�
#absolute_difference/ones_like/ShapeConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB 
�
#absolute_difference/ones_like/ConstConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *  �?
�
absolute_difference/ones_likeFill#absolute_difference/ones_like/Shape#absolute_difference/ones_like/Const*
T0*

index_type0
�
absolute_difference/SelectSelectabsolute_difference/Equalabsolute_difference/ones_likeabsolute_difference/num_present*
T0
b
absolute_difference/divRealDivabsolute_difference/Sum_1absolute_difference/Select*
T0
�
absolute_difference/zeros_likeConstC^absolute_difference/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
�
absolute_difference/valueSelectabsolute_difference/Greaterabsolute_difference/divabsolute_difference/zeros_like*
T0 