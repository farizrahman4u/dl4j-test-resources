
i
VariableConst*
dtype0*I
value@B>"0              �?      �?                  �?    
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
e
in_1Const*
dtype0*I
value@B>"0���X��>�ÿ>L+�DAF?��>��S�D�e?���
%?��(��٪>
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
A
in_2Const*
dtype0*%
valueB"            
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
O
hinge_loss/ones_like/ShapeConst*
dtype0*
valueB"      
G
hinge_loss/ones_like/ConstConst*
dtype0*
valueB
 *  �?
o
hinge_loss/ones_likeFillhinge_loss/ones_like/Shapehinge_loss/ones_like/Const*
T0*

index_type0
=
hinge_loss/mul/xConst*
dtype0*
valueB
 *   @
?
hinge_loss/mulMulhinge_loss/mul/xVariable/read*
T0
D
hinge_loss/SubSubhinge_loss/mulhinge_loss/ones_like*
T0
;
hinge_loss/Mul_1Mulhinge_loss/Sub	in_1/read*
T0
H
hinge_loss/Sub_1Subhinge_loss/ones_likehinge_loss/Mul_1*
T0
2
hinge_loss/ReluReluhinge_loss/Sub_1*
T0
A
9hinge_loss/assert_broadcastable/static_dims_check_successNoOp
x
hinge_loss/Mul_2Mulhinge_loss/Relu	in_2/read:^hinge_loss/assert_broadcastable/static_dims_check_success*
T0
�
hinge_loss/ConstConst:^hinge_loss/assert_broadcastable/static_dims_check_success*
dtype0*
valueB"       
_
hinge_loss/SumSumhinge_loss/Mul_2hinge_loss/Const*
T0*

Tidx0*
	keep_dims(  