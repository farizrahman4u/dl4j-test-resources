
C
inputPlaceholder*
dtype0* 
shape:���������
i
VariableConst*
dtype0*I
value@B>"0��xk���?�{r���|e���;�����0	��N����?m�}����?
I
Variable/readIdentityVariable*
_class
loc:@Variable*
T0
G

Variable_1Const*
dtype0*%
valueB"o܃�~����C�?
O
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0
8
unstackUnpackinput*	
num*
T0*

axis
4
	rnn/ShapeShapeunstack*
out_type0*
T0
E
rnn/strided_slice/stackConst*
dtype0*
valueB: 
G
rnn/strided_slice/stack_1Const*
dtype0*
valueB:
G
rnn/strided_slice/stack_2Const*
dtype0*
valueB:
�
rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
new_axis_mask *
Index0*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
S
)rnn/BasicLSTMCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : 
�
%rnn/BasicLSTMCellZeroState/ExpandDims
ExpandDimsrnn/strided_slice)rnn/BasicLSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0
N
 rnn/BasicLSTMCellZeroState/ConstConst*
dtype0*
valueB:
P
&rnn/BasicLSTMCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
!rnn/BasicLSTMCellZeroState/concatConcatV2%rnn/BasicLSTMCellZeroState/ExpandDims rnn/BasicLSTMCellZeroState/Const&rnn/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
N
W
&rnn/BasicLSTMCellZeroState/zeros/ConstConst*
dtype0*
valueB 2        
|
 rnn/BasicLSTMCellZeroState/zerosFill!rnn/BasicLSTMCellZeroState/concat&rnn/BasicLSTMCellZeroState/zeros/Const*
T0
U
+rnn/BasicLSTMCellZeroState/ExpandDims_2/dimConst*
dtype0*
value	B : 
�
'rnn/BasicLSTMCellZeroState/ExpandDims_2
ExpandDimsrnn/strided_slice+rnn/BasicLSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0
P
"rnn/BasicLSTMCellZeroState/Const_2Const*
dtype0*
valueB:
R
(rnn/BasicLSTMCellZeroState/concat_1/axisConst*
dtype0*
value	B : 
�
#rnn/BasicLSTMCellZeroState/concat_1ConcatV2'rnn/BasicLSTMCellZeroState/ExpandDims_2"rnn/BasicLSTMCellZeroState/Const_2(rnn/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N
Y
(rnn/BasicLSTMCellZeroState/zeros_1/ConstConst*
dtype0*
valueB 2        
�
"rnn/BasicLSTMCellZeroState/zeros_1Fill#rnn/BasicLSTMCellZeroState/concat_1(rnn/BasicLSTMCellZeroState/zeros_1/Const*
T0
�
rnn/basic_lstm_cell/kernelConst*
dtype0*�
value�B�"����9AۿayE�@ȿ�(� ��?R�8��ٿ�m���?��qT��v��8�ӿ�biO�ƿL�����?�	����?� ��_X׿����g�E�Rf�	k�ÿ.�}u ��?mP�hD�?Z�ˑ�g��Đ�����?:�63�Ͽ���6���?w�9��>�?0tt�x׿��l�(�ȿRE����?>��y�B��(yPG��?�Q�y�?�LE���ֿkA�ڿ7�M+��?��!)�?������Od	*���?�`x���?̘�t�߿Cu��E�?�s:w^������(��?Y�Cp:U�?kQ���ݿ�A�]���񠴂j�?�Y?��῞H���ݿ�N�u�2ѿ��br��ۿۜ�l���QF�e��?�-����<���?08��6�?X`��\�?h�'�]��?BS����?�"0H�f�?���n�����Ձ�?��R���?.��Wy��?��m���?
P
rnn/basic_lstm_cell/kernel/readIdentityrnn/basic_lstm_cell/kernel*
T0
]
3rnn/rnn/basic_lstm_cell/basic_lstm_cell/concat/axisConst*
dtype0*
value	B :
�
.rnn/rnn/basic_lstm_cell/basic_lstm_cell/concatConcatV2unstack"rnn/BasicLSTMCellZeroState/zeros_13rnn/rnn/basic_lstm_cell/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N
�
.rnn/rnn/basic_lstm_cell/basic_lstm_cell/MatMulMatMul.rnn/rnn/basic_lstm_cell/basic_lstm_cell/concatrnn/basic_lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0
�
rnn/basic_lstm_cell/biasConst*
dtype0*u
valuelBj"`"\o�/Q��mG�/W"�j���o�}/LK�?��[>�p����^��p�I�zH�K�_��8�6��bC�M
H��N�`ul.#�頜3��#�
L
rnn/basic_lstm_cell/bias/readIdentityrnn/basic_lstm_cell/bias*
T0
�
/rnn/rnn/basic_lstm_cell/basic_lstm_cell/BiasAddBiasAdd.rnn/rnn/basic_lstm_cell/basic_lstm_cell/MatMulrnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC
Q
'rnn/rnn/basic_lstm_cell/split/split_dimConst*
dtype0*
value	B :
�
rnn/rnn/basic_lstm_cell/splitSplit'rnn/rnn/basic_lstm_cell/split/split_dim/rnn/rnn/basic_lstm_cell/basic_lstm_cell/BiasAdd*
	num_split*
T0
N
rnn/rnn/basic_lstm_cell/add/yConst*
dtype0*
valueB 2      �?
k
rnn/rnn/basic_lstm_cell/addAddrnn/rnn/basic_lstm_cell/split:2rnn/rnn/basic_lstm_cell/add/y*
T0
P
rnn/rnn/basic_lstm_cell/SigmoidSigmoidrnn/rnn/basic_lstm_cell/add*
T0
n
rnn/rnn/basic_lstm_cell/mulMul rnn/BasicLSTMCellZeroState/zerosrnn/rnn/basic_lstm_cell/Sigmoid*
T0
T
!rnn/rnn/basic_lstm_cell/Sigmoid_1Sigmoidrnn/rnn/basic_lstm_cell/split*
T0
N
rnn/rnn/basic_lstm_cell/TanhTanhrnn/rnn/basic_lstm_cell/split:1*
T0
n
rnn/rnn/basic_lstm_cell/mul_1Mul!rnn/rnn/basic_lstm_cell/Sigmoid_1rnn/rnn/basic_lstm_cell/Tanh*
T0
i
rnn/rnn/basic_lstm_cell/add_1Addrnn/rnn/basic_lstm_cell/mulrnn/rnn/basic_lstm_cell/mul_1*
T0
N
rnn/rnn/basic_lstm_cell/Tanh_1Tanhrnn/rnn/basic_lstm_cell/add_1*
T0
V
!rnn/rnn/basic_lstm_cell/Sigmoid_2Sigmoidrnn/rnn/basic_lstm_cell/split:3*
T0
p
rnn/rnn/basic_lstm_cell/mul_2Mulrnn/rnn/basic_lstm_cell/Tanh_1!rnn/rnn/basic_lstm_cell/Sigmoid_2*
T0
_
5rnn/rnn/basic_lstm_cell/basic_lstm_cell_1/concat/axisConst*
dtype0*
value	B :
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_1/concatConcatV2	unstack:1rnn/rnn/basic_lstm_cell/mul_25rnn/rnn/basic_lstm_cell/basic_lstm_cell_1/concat/axis*

Tidx0*
T0*
N
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_1/MatMulMatMul0rnn/rnn/basic_lstm_cell/basic_lstm_cell_1/concatrnn/basic_lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0
�
1rnn/rnn/basic_lstm_cell/basic_lstm_cell_1/BiasAddBiasAdd0rnn/rnn/basic_lstm_cell/basic_lstm_cell_1/MatMulrnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC
S
)rnn/rnn/basic_lstm_cell/split_1/split_dimConst*
dtype0*
value	B :
�
rnn/rnn/basic_lstm_cell/split_1Split)rnn/rnn/basic_lstm_cell/split_1/split_dim1rnn/rnn/basic_lstm_cell/basic_lstm_cell_1/BiasAdd*
	num_split*
T0
P
rnn/rnn/basic_lstm_cell/add_2/yConst*
dtype0*
valueB 2      �?
q
rnn/rnn/basic_lstm_cell/add_2Add!rnn/rnn/basic_lstm_cell/split_1:2rnn/rnn/basic_lstm_cell/add_2/y*
T0
T
!rnn/rnn/basic_lstm_cell/Sigmoid_3Sigmoidrnn/rnn/basic_lstm_cell/add_2*
T0
o
rnn/rnn/basic_lstm_cell/mul_3Mulrnn/rnn/basic_lstm_cell/add_1!rnn/rnn/basic_lstm_cell/Sigmoid_3*
T0
V
!rnn/rnn/basic_lstm_cell/Sigmoid_4Sigmoidrnn/rnn/basic_lstm_cell/split_1*
T0
R
rnn/rnn/basic_lstm_cell/Tanh_2Tanh!rnn/rnn/basic_lstm_cell/split_1:1*
T0
p
rnn/rnn/basic_lstm_cell/mul_4Mul!rnn/rnn/basic_lstm_cell/Sigmoid_4rnn/rnn/basic_lstm_cell/Tanh_2*
T0
k
rnn/rnn/basic_lstm_cell/add_3Addrnn/rnn/basic_lstm_cell/mul_3rnn/rnn/basic_lstm_cell/mul_4*
T0
N
rnn/rnn/basic_lstm_cell/Tanh_3Tanhrnn/rnn/basic_lstm_cell/add_3*
T0
X
!rnn/rnn/basic_lstm_cell/Sigmoid_5Sigmoid!rnn/rnn/basic_lstm_cell/split_1:3*
T0
p
rnn/rnn/basic_lstm_cell/mul_5Mulrnn/rnn/basic_lstm_cell/Tanh_3!rnn/rnn/basic_lstm_cell/Sigmoid_5*
T0
_
5rnn/rnn/basic_lstm_cell/basic_lstm_cell_2/concat/axisConst*
dtype0*
value	B :
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_2/concatConcatV2	unstack:2rnn/rnn/basic_lstm_cell/mul_55rnn/rnn/basic_lstm_cell/basic_lstm_cell_2/concat/axis*

Tidx0*
T0*
N
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_2/MatMulMatMul0rnn/rnn/basic_lstm_cell/basic_lstm_cell_2/concatrnn/basic_lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0
�
1rnn/rnn/basic_lstm_cell/basic_lstm_cell_2/BiasAddBiasAdd0rnn/rnn/basic_lstm_cell/basic_lstm_cell_2/MatMulrnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC
S
)rnn/rnn/basic_lstm_cell/split_2/split_dimConst*
dtype0*
value	B :
�
rnn/rnn/basic_lstm_cell/split_2Split)rnn/rnn/basic_lstm_cell/split_2/split_dim1rnn/rnn/basic_lstm_cell/basic_lstm_cell_2/BiasAdd*
	num_split*
T0
P
rnn/rnn/basic_lstm_cell/add_4/yConst*
dtype0*
valueB 2      �?
q
rnn/rnn/basic_lstm_cell/add_4Add!rnn/rnn/basic_lstm_cell/split_2:2rnn/rnn/basic_lstm_cell/add_4/y*
T0
T
!rnn/rnn/basic_lstm_cell/Sigmoid_6Sigmoidrnn/rnn/basic_lstm_cell/add_4*
T0
o
rnn/rnn/basic_lstm_cell/mul_6Mulrnn/rnn/basic_lstm_cell/add_3!rnn/rnn/basic_lstm_cell/Sigmoid_6*
T0
V
!rnn/rnn/basic_lstm_cell/Sigmoid_7Sigmoidrnn/rnn/basic_lstm_cell/split_2*
T0
R
rnn/rnn/basic_lstm_cell/Tanh_4Tanh!rnn/rnn/basic_lstm_cell/split_2:1*
T0
p
rnn/rnn/basic_lstm_cell/mul_7Mul!rnn/rnn/basic_lstm_cell/Sigmoid_7rnn/rnn/basic_lstm_cell/Tanh_4*
T0
k
rnn/rnn/basic_lstm_cell/add_5Addrnn/rnn/basic_lstm_cell/mul_6rnn/rnn/basic_lstm_cell/mul_7*
T0
N
rnn/rnn/basic_lstm_cell/Tanh_5Tanhrnn/rnn/basic_lstm_cell/add_5*
T0
X
!rnn/rnn/basic_lstm_cell/Sigmoid_8Sigmoid!rnn/rnn/basic_lstm_cell/split_2:3*
T0
p
rnn/rnn/basic_lstm_cell/mul_8Mulrnn/rnn/basic_lstm_cell/Tanh_5!rnn/rnn/basic_lstm_cell/Sigmoid_8*
T0
_
5rnn/rnn/basic_lstm_cell/basic_lstm_cell_3/concat/axisConst*
dtype0*
value	B :
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_3/concatConcatV2	unstack:3rnn/rnn/basic_lstm_cell/mul_85rnn/rnn/basic_lstm_cell/basic_lstm_cell_3/concat/axis*

Tidx0*
T0*
N
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_3/MatMulMatMul0rnn/rnn/basic_lstm_cell/basic_lstm_cell_3/concatrnn/basic_lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0
�
1rnn/rnn/basic_lstm_cell/basic_lstm_cell_3/BiasAddBiasAdd0rnn/rnn/basic_lstm_cell/basic_lstm_cell_3/MatMulrnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC
S
)rnn/rnn/basic_lstm_cell/split_3/split_dimConst*
dtype0*
value	B :
�
rnn/rnn/basic_lstm_cell/split_3Split)rnn/rnn/basic_lstm_cell/split_3/split_dim1rnn/rnn/basic_lstm_cell/basic_lstm_cell_3/BiasAdd*
	num_split*
T0
P
rnn/rnn/basic_lstm_cell/add_6/yConst*
dtype0*
valueB 2      �?
q
rnn/rnn/basic_lstm_cell/add_6Add!rnn/rnn/basic_lstm_cell/split_3:2rnn/rnn/basic_lstm_cell/add_6/y*
T0
T
!rnn/rnn/basic_lstm_cell/Sigmoid_9Sigmoidrnn/rnn/basic_lstm_cell/add_6*
T0
o
rnn/rnn/basic_lstm_cell/mul_9Mulrnn/rnn/basic_lstm_cell/add_5!rnn/rnn/basic_lstm_cell/Sigmoid_9*
T0
W
"rnn/rnn/basic_lstm_cell/Sigmoid_10Sigmoidrnn/rnn/basic_lstm_cell/split_3*
T0
R
rnn/rnn/basic_lstm_cell/Tanh_6Tanh!rnn/rnn/basic_lstm_cell/split_3:1*
T0
r
rnn/rnn/basic_lstm_cell/mul_10Mul"rnn/rnn/basic_lstm_cell/Sigmoid_10rnn/rnn/basic_lstm_cell/Tanh_6*
T0
l
rnn/rnn/basic_lstm_cell/add_7Addrnn/rnn/basic_lstm_cell/mul_9rnn/rnn/basic_lstm_cell/mul_10*
T0
N
rnn/rnn/basic_lstm_cell/Tanh_7Tanhrnn/rnn/basic_lstm_cell/add_7*
T0
Y
"rnn/rnn/basic_lstm_cell/Sigmoid_11Sigmoid!rnn/rnn/basic_lstm_cell/split_3:3*
T0
r
rnn/rnn/basic_lstm_cell/mul_11Mulrnn/rnn/basic_lstm_cell/Tanh_7"rnn/rnn/basic_lstm_cell/Sigmoid_11*
T0
_
5rnn/rnn/basic_lstm_cell/basic_lstm_cell_4/concat/axisConst*
dtype0*
value	B :
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_4/concatConcatV2	unstack:4rnn/rnn/basic_lstm_cell/mul_115rnn/rnn/basic_lstm_cell/basic_lstm_cell_4/concat/axis*

Tidx0*
T0*
N
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_4/MatMulMatMul0rnn/rnn/basic_lstm_cell/basic_lstm_cell_4/concatrnn/basic_lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0
�
1rnn/rnn/basic_lstm_cell/basic_lstm_cell_4/BiasAddBiasAdd0rnn/rnn/basic_lstm_cell/basic_lstm_cell_4/MatMulrnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC
S
)rnn/rnn/basic_lstm_cell/split_4/split_dimConst*
dtype0*
value	B :
�
rnn/rnn/basic_lstm_cell/split_4Split)rnn/rnn/basic_lstm_cell/split_4/split_dim1rnn/rnn/basic_lstm_cell/basic_lstm_cell_4/BiasAdd*
	num_split*
T0
P
rnn/rnn/basic_lstm_cell/add_8/yConst*
dtype0*
valueB 2      �?
q
rnn/rnn/basic_lstm_cell/add_8Add!rnn/rnn/basic_lstm_cell/split_4:2rnn/rnn/basic_lstm_cell/add_8/y*
T0
U
"rnn/rnn/basic_lstm_cell/Sigmoid_12Sigmoidrnn/rnn/basic_lstm_cell/add_8*
T0
q
rnn/rnn/basic_lstm_cell/mul_12Mulrnn/rnn/basic_lstm_cell/add_7"rnn/rnn/basic_lstm_cell/Sigmoid_12*
T0
W
"rnn/rnn/basic_lstm_cell/Sigmoid_13Sigmoidrnn/rnn/basic_lstm_cell/split_4*
T0
R
rnn/rnn/basic_lstm_cell/Tanh_8Tanh!rnn/rnn/basic_lstm_cell/split_4:1*
T0
r
rnn/rnn/basic_lstm_cell/mul_13Mul"rnn/rnn/basic_lstm_cell/Sigmoid_13rnn/rnn/basic_lstm_cell/Tanh_8*
T0
m
rnn/rnn/basic_lstm_cell/add_9Addrnn/rnn/basic_lstm_cell/mul_12rnn/rnn/basic_lstm_cell/mul_13*
T0
N
rnn/rnn/basic_lstm_cell/Tanh_9Tanhrnn/rnn/basic_lstm_cell/add_9*
T0
Y
"rnn/rnn/basic_lstm_cell/Sigmoid_14Sigmoid!rnn/rnn/basic_lstm_cell/split_4:3*
T0
r
rnn/rnn/basic_lstm_cell/mul_14Mulrnn/rnn/basic_lstm_cell/Tanh_9"rnn/rnn/basic_lstm_cell/Sigmoid_14*
T0
_
5rnn/rnn/basic_lstm_cell/basic_lstm_cell_5/concat/axisConst*
dtype0*
value	B :
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_5/concatConcatV2	unstack:5rnn/rnn/basic_lstm_cell/mul_145rnn/rnn/basic_lstm_cell/basic_lstm_cell_5/concat/axis*

Tidx0*
T0*
N
�
0rnn/rnn/basic_lstm_cell/basic_lstm_cell_5/MatMulMatMul0rnn/rnn/basic_lstm_cell/basic_lstm_cell_5/concatrnn/basic_lstm_cell/kernel/read*
transpose_b( *
transpose_a( *
T0
�
1rnn/rnn/basic_lstm_cell/basic_lstm_cell_5/BiasAddBiasAdd0rnn/rnn/basic_lstm_cell/basic_lstm_cell_5/MatMulrnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC
S
)rnn/rnn/basic_lstm_cell/split_5/split_dimConst*
dtype0*
value	B :
�
rnn/rnn/basic_lstm_cell/split_5Split)rnn/rnn/basic_lstm_cell/split_5/split_dim1rnn/rnn/basic_lstm_cell/basic_lstm_cell_5/BiasAdd*
	num_split*
T0
Q
 rnn/rnn/basic_lstm_cell/add_10/yConst*
dtype0*
valueB 2      �?
s
rnn/rnn/basic_lstm_cell/add_10Add!rnn/rnn/basic_lstm_cell/split_5:2 rnn/rnn/basic_lstm_cell/add_10/y*
T0
V
"rnn/rnn/basic_lstm_cell/Sigmoid_15Sigmoidrnn/rnn/basic_lstm_cell/add_10*
T0
q
rnn/rnn/basic_lstm_cell/mul_15Mulrnn/rnn/basic_lstm_cell/add_9"rnn/rnn/basic_lstm_cell/Sigmoid_15*
T0
W
"rnn/rnn/basic_lstm_cell/Sigmoid_16Sigmoidrnn/rnn/basic_lstm_cell/split_5*
T0
S
rnn/rnn/basic_lstm_cell/Tanh_10Tanh!rnn/rnn/basic_lstm_cell/split_5:1*
T0
s
rnn/rnn/basic_lstm_cell/mul_16Mul"rnn/rnn/basic_lstm_cell/Sigmoid_16rnn/rnn/basic_lstm_cell/Tanh_10*
T0
n
rnn/rnn/basic_lstm_cell/add_11Addrnn/rnn/basic_lstm_cell/mul_15rnn/rnn/basic_lstm_cell/mul_16*
T0
P
rnn/rnn/basic_lstm_cell/Tanh_11Tanhrnn/rnn/basic_lstm_cell/add_11*
T0
Y
"rnn/rnn/basic_lstm_cell/Sigmoid_17Sigmoid!rnn/rnn/basic_lstm_cell/split_5:3*
T0
s
rnn/rnn/basic_lstm_cell/mul_17Mulrnn/rnn/basic_lstm_cell/Tanh_11"rnn/rnn/basic_lstm_cell/Sigmoid_17*
T0
n
MatMulMatMulrnn/rnn/basic_lstm_cell/mul_17Variable/read*
transpose_b( *
transpose_a( *
T0
,
addAddMatMulVariable_1/read*
T0
 
outputIdentityadd*
T0 