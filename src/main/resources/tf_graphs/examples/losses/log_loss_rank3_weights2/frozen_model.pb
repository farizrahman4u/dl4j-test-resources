
�
in_0Const*
dtype0*}
valuetBr"`~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?�+-?HM8>v�,?p�e>
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
�
in_1Const*
dtype0*}
valuetBr"`�E?��m?�|?ز�>��$?@�?�n&?��B?ܰB?��>ps?�*\?`�I?��d?w�>�77?h+�>t7N?�Ru?��?*kU? �'<�l�>`�>
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
I
in_2Const*
dtype0*-
value$B""��q?�~?�]?�O�>
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
;
log_loss/add/yConst*
dtype0*
valueB
 *���3
7
log_loss/addAdd	in_1/readlog_loss/add/y*
T0
*
log_loss/LogLoglog_loss/add*
T0
5
log_loss/MulMul	in_0/readlog_loss/Log*
T0
*
log_loss/NegNeglog_loss/Mul*
T0
;
log_loss/sub/xConst*
dtype0*
valueB
 *  �?
7
log_loss/subSublog_loss/sub/x	in_0/read*
T0
=
log_loss/sub_1/xConst*
dtype0*
valueB
 *  �?
;
log_loss/sub_1Sublog_loss/sub_1/x	in_1/read*
T0
=
log_loss/add_1/yConst*
dtype0*
valueB
 *���3
@
log_loss/add_1Addlog_loss/sub_1log_loss/add_1/y*
T0
.
log_loss/Log_1Loglog_loss/add_1*
T0
<
log_loss/Mul_1Mullog_loss/sublog_loss/Log_1*
T0
<
log_loss/sub_2Sublog_loss/Neglog_loss/Mul_1*
T0
?
7log_loss/assert_broadcastable/static_dims_check_successNoOp
s
log_loss/Mul_2Mullog_loss/sub_2	in_2/read8^log_loss/assert_broadcastable/static_dims_check_success*
T0
�
log_loss/ConstConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*!
valueB"          
Y
log_loss/SumSumlog_loss/Mul_2log_loss/Const*
T0*

Tidx0*
	keep_dims( 
�
log_loss/num_present/Equal/yConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
U
log_loss/num_present/EqualEqual	in_2/readlog_loss/num_present/Equal/y*
T0
�
log_loss/num_present/zeros_likeConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*!
valueB*    
�
$log_loss/num_present/ones_like/ShapeConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*!
valueB"         
�
$log_loss/num_present/ones_like/ConstConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *  �?
�
log_loss/num_present/ones_likeFill$log_loss/num_present/ones_like/Shape$log_loss/num_present/ones_like/Const*
T0*

index_type0
�
log_loss/num_present/SelectSelectlog_loss/num_present/Equallog_loss/num_present/zeros_likelog_loss/num_present/ones_like*
T0
�
Ulog_loss/num_present/broadcast_weights/assert_broadcastable/static_dims_check_successNoOp8^log_loss/assert_broadcastable/static_dims_check_success
�
6log_loss/num_present/broadcast_weights/ones_like/ShapeConst8^log_loss/assert_broadcastable/static_dims_check_successV^log_loss/num_present/broadcast_weights/assert_broadcastable/static_dims_check_success*
dtype0*!
valueB"         
�
6log_loss/num_present/broadcast_weights/ones_like/ConstConst8^log_loss/assert_broadcastable/static_dims_check_successV^log_loss/num_present/broadcast_weights/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *  �?
�
0log_loss/num_present/broadcast_weights/ones_likeFill6log_loss/num_present/broadcast_weights/ones_like/Shape6log_loss/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0
�
&log_loss/num_present/broadcast_weightsMullog_loss/num_present/Select0log_loss/num_present/broadcast_weights/ones_like*
T0
�
log_loss/num_present/ConstConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*!
valueB"          
�
log_loss/num_presentSum&log_loss/num_present/broadcast_weightslog_loss/num_present/Const*
T0*

Tidx0*
	keep_dims( 
s
log_loss/Const_1Const8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB 
[
log_loss/Sum_1Sumlog_loss/Sumlog_loss/Const_1*
T0*

Tidx0*
	keep_dims( 
y
log_loss/Greater/yConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
N
log_loss/GreaterGreaterlog_loss/num_presentlog_loss/Greater/y*
T0
w
log_loss/Equal/yConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
H
log_loss/EqualEquallog_loss/num_presentlog_loss/Equal/y*
T0
{
log_loss/ones_like/ShapeConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB 

log_loss/ones_like/ConstConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *  �?
i
log_loss/ones_likeFilllog_loss/ones_like/Shapelog_loss/ones_like/Const*
T0*

index_type0
\
log_loss/SelectSelectlog_loss/Equallog_loss/ones_likelog_loss/num_present*
T0
A
log_loss/divRealDivlog_loss/Sum_1log_loss/Select*
T0
z
log_loss/zeros_likeConst8^log_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB
 *    
V
log_loss/valueSelectlog_loss/Greaterlog_loss/divlog_loss/zeros_like*
T0 