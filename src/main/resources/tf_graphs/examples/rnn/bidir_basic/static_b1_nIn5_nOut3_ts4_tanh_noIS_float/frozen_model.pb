
�
in_0Const*m
valuedBb"P~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?*
dtype0
=
	in_0/readIdentityin_0*
_class
	loc:@in_0*
T0
<
unstackUnpack	in_0/read*	
num*
T0*

axis
a
3bidirectional_rnn/fw/fw/BasicRNNCellZeroState/ConstConst*
valueB:*
dtype0
c
5bidirectional_rnn/fw/fw/BasicRNNCellZeroState/Const_1Const*
valueB:*
dtype0
c
9bidirectional_rnn/fw/fw/BasicRNNCellZeroState/concat/axisConst*
value	B : *
dtype0
�
4bidirectional_rnn/fw/fw/BasicRNNCellZeroState/concatConcatV23bidirectional_rnn/fw/fw/BasicRNNCellZeroState/Const5bidirectional_rnn/fw/fw/BasicRNNCellZeroState/Const_19bidirectional_rnn/fw/fw/BasicRNNCellZeroState/concat/axis*
T0*
N*

Tidx0
f
9bidirectional_rnn/fw/fw/BasicRNNCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
�
3bidirectional_rnn/fw/fw/BasicRNNCellZeroState/zerosFill4bidirectional_rnn/fw/fw/BasicRNNCellZeroState/concat9bidirectional_rnn/fw/fw/BasicRNNCellZeroState/zeros/Const*
T0*

index_type0
�
*bidirectional_rnn/fw/basic_rnn_cell/kernelConst*y
valuepBn"`�͏�IW����(?A>>\L���ܾ}�%?����D꾰6�H���2�m@?��>�q�>s�6?(�wg>y7�Q3?����n�6�*
dtype0
p
/bidirectional_rnn/fw/basic_rnn_cell/kernel/readIdentity*bidirectional_rnn/fw/basic_rnn_cell/kernel*
T0
a
(bidirectional_rnn/fw/basic_rnn_cell/biasConst*!
valueB"            *
dtype0
l
-bidirectional_rnn/fw/basic_rnn_cell/bias/readIdentity(bidirectional_rnn/fw/basic_rnn_cell/bias*
T0
\
2bidirectional_rnn/fw/fw/basic_rnn_cell/concat/axisConst*
value	B :*
dtype0
�
-bidirectional_rnn/fw/fw/basic_rnn_cell/concatConcatV2unstack3bidirectional_rnn/fw/fw/BasicRNNCellZeroState/zeros2bidirectional_rnn/fw/fw/basic_rnn_cell/concat/axis*
N*

Tidx0*
T0
�
-bidirectional_rnn/fw/fw/basic_rnn_cell/MatMulMatMul-bidirectional_rnn/fw/fw/basic_rnn_cell/concat/bidirectional_rnn/fw/basic_rnn_cell/kernel/read*
T0*
transpose_a( *
transpose_b( 
�
.bidirectional_rnn/fw/fw/basic_rnn_cell/BiasAddBiasAdd-bidirectional_rnn/fw/fw/basic_rnn_cell/MatMul-bidirectional_rnn/fw/basic_rnn_cell/bias/read*
T0*
data_formatNHWC
l
+bidirectional_rnn/fw/fw/basic_rnn_cell/TanhTanh.bidirectional_rnn/fw/fw/basic_rnn_cell/BiasAdd*
T0
^
4bidirectional_rnn/fw/fw/basic_rnn_cell/concat_1/axisConst*
value	B :*
dtype0
�
/bidirectional_rnn/fw/fw/basic_rnn_cell/concat_1ConcatV2	unstack:1+bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh4bidirectional_rnn/fw/fw/basic_rnn_cell/concat_1/axis*
T0*
N*

Tidx0
�
/bidirectional_rnn/fw/fw/basic_rnn_cell/MatMul_1MatMul/bidirectional_rnn/fw/fw/basic_rnn_cell/concat_1/bidirectional_rnn/fw/basic_rnn_cell/kernel/read*
transpose_a( *
transpose_b( *
T0
�
0bidirectional_rnn/fw/fw/basic_rnn_cell/BiasAdd_1BiasAdd/bidirectional_rnn/fw/fw/basic_rnn_cell/MatMul_1-bidirectional_rnn/fw/basic_rnn_cell/bias/read*
data_formatNHWC*
T0
p
-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_1Tanh0bidirectional_rnn/fw/fw/basic_rnn_cell/BiasAdd_1*
T0
^
4bidirectional_rnn/fw/fw/basic_rnn_cell/concat_2/axisConst*
value	B :*
dtype0
�
/bidirectional_rnn/fw/fw/basic_rnn_cell/concat_2ConcatV2	unstack:2-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_14bidirectional_rnn/fw/fw/basic_rnn_cell/concat_2/axis*
N*

Tidx0*
T0
�
/bidirectional_rnn/fw/fw/basic_rnn_cell/MatMul_2MatMul/bidirectional_rnn/fw/fw/basic_rnn_cell/concat_2/bidirectional_rnn/fw/basic_rnn_cell/kernel/read*
transpose_b( *
T0*
transpose_a( 
�
0bidirectional_rnn/fw/fw/basic_rnn_cell/BiasAdd_2BiasAdd/bidirectional_rnn/fw/fw/basic_rnn_cell/MatMul_2-bidirectional_rnn/fw/basic_rnn_cell/bias/read*
data_formatNHWC*
T0
p
-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_2Tanh0bidirectional_rnn/fw/fw/basic_rnn_cell/BiasAdd_2*
T0
^
4bidirectional_rnn/fw/fw/basic_rnn_cell/concat_3/axisConst*
value	B :*
dtype0
�
/bidirectional_rnn/fw/fw/basic_rnn_cell/concat_3ConcatV2	unstack:3-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_24bidirectional_rnn/fw/fw/basic_rnn_cell/concat_3/axis*

Tidx0*
T0*
N
�
/bidirectional_rnn/fw/fw/basic_rnn_cell/MatMul_3MatMul/bidirectional_rnn/fw/fw/basic_rnn_cell/concat_3/bidirectional_rnn/fw/basic_rnn_cell/kernel/read*
transpose_a( *
transpose_b( *
T0
�
0bidirectional_rnn/fw/fw/basic_rnn_cell/BiasAdd_3BiasAdd/bidirectional_rnn/fw/fw/basic_rnn_cell/MatMul_3-bidirectional_rnn/fw/basic_rnn_cell/bias/read*
T0*
data_formatNHWC
p
-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_3Tanh0bidirectional_rnn/fw/fw/basic_rnn_cell/BiasAdd_3*
T0
a
3bidirectional_rnn/bw/bw/BasicRNNCellZeroState/ConstConst*
dtype0*
valueB:
c
5bidirectional_rnn/bw/bw/BasicRNNCellZeroState/Const_1Const*
valueB:*
dtype0
c
9bidirectional_rnn/bw/bw/BasicRNNCellZeroState/concat/axisConst*
value	B : *
dtype0
�
4bidirectional_rnn/bw/bw/BasicRNNCellZeroState/concatConcatV23bidirectional_rnn/bw/bw/BasicRNNCellZeroState/Const5bidirectional_rnn/bw/bw/BasicRNNCellZeroState/Const_19bidirectional_rnn/bw/bw/BasicRNNCellZeroState/concat/axis*
T0*
N*

Tidx0
f
9bidirectional_rnn/bw/bw/BasicRNNCellZeroState/zeros/ConstConst*
dtype0*
valueB
 *    
�
3bidirectional_rnn/bw/bw/BasicRNNCellZeroState/zerosFill4bidirectional_rnn/bw/bw/BasicRNNCellZeroState/concat9bidirectional_rnn/bw/bw/BasicRNNCellZeroState/zeros/Const*

index_type0*
T0
�
*bidirectional_rnn/bw/basic_rnn_cell/kernelConst*y
valuepBn"`d�Z>�
d=�d���Ҿ-:?�8�@���g�0?]/:?`lq=ol�)R�S~�(�˽�ƿ���ʾF�ʾP�.�҆�> �$9j>�>�*��5���>*
dtype0
p
/bidirectional_rnn/bw/basic_rnn_cell/kernel/readIdentity*bidirectional_rnn/bw/basic_rnn_cell/kernel*
T0
a
(bidirectional_rnn/bw/basic_rnn_cell/biasConst*!
valueB"            *
dtype0
l
-bidirectional_rnn/bw/basic_rnn_cell/bias/readIdentity(bidirectional_rnn/bw/basic_rnn_cell/bias*
T0
\
2bidirectional_rnn/bw/bw/basic_rnn_cell/concat/axisConst*
value	B :*
dtype0
�
-bidirectional_rnn/bw/bw/basic_rnn_cell/concatConcatV2	unstack:33bidirectional_rnn/bw/bw/BasicRNNCellZeroState/zeros2bidirectional_rnn/bw/bw/basic_rnn_cell/concat/axis*

Tidx0*
T0*
N
�
-bidirectional_rnn/bw/bw/basic_rnn_cell/MatMulMatMul-bidirectional_rnn/bw/bw/basic_rnn_cell/concat/bidirectional_rnn/bw/basic_rnn_cell/kernel/read*
T0*
transpose_a( *
transpose_b( 
�
.bidirectional_rnn/bw/bw/basic_rnn_cell/BiasAddBiasAdd-bidirectional_rnn/bw/bw/basic_rnn_cell/MatMul-bidirectional_rnn/bw/basic_rnn_cell/bias/read*
T0*
data_formatNHWC
l
+bidirectional_rnn/bw/bw/basic_rnn_cell/TanhTanh.bidirectional_rnn/bw/bw/basic_rnn_cell/BiasAdd*
T0
^
4bidirectional_rnn/bw/bw/basic_rnn_cell/concat_1/axisConst*
value	B :*
dtype0
�
/bidirectional_rnn/bw/bw/basic_rnn_cell/concat_1ConcatV2	unstack:2+bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh4bidirectional_rnn/bw/bw/basic_rnn_cell/concat_1/axis*
N*

Tidx0*
T0
�
/bidirectional_rnn/bw/bw/basic_rnn_cell/MatMul_1MatMul/bidirectional_rnn/bw/bw/basic_rnn_cell/concat_1/bidirectional_rnn/bw/basic_rnn_cell/kernel/read*
T0*
transpose_a( *
transpose_b( 
�
0bidirectional_rnn/bw/bw/basic_rnn_cell/BiasAdd_1BiasAdd/bidirectional_rnn/bw/bw/basic_rnn_cell/MatMul_1-bidirectional_rnn/bw/basic_rnn_cell/bias/read*
T0*
data_formatNHWC
p
-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_1Tanh0bidirectional_rnn/bw/bw/basic_rnn_cell/BiasAdd_1*
T0
^
4bidirectional_rnn/bw/bw/basic_rnn_cell/concat_2/axisConst*
value	B :*
dtype0
�
/bidirectional_rnn/bw/bw/basic_rnn_cell/concat_2ConcatV2	unstack:1-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_14bidirectional_rnn/bw/bw/basic_rnn_cell/concat_2/axis*

Tidx0*
T0*
N
�
/bidirectional_rnn/bw/bw/basic_rnn_cell/MatMul_2MatMul/bidirectional_rnn/bw/bw/basic_rnn_cell/concat_2/bidirectional_rnn/bw/basic_rnn_cell/kernel/read*
T0*
transpose_a( *
transpose_b( 
�
0bidirectional_rnn/bw/bw/basic_rnn_cell/BiasAdd_2BiasAdd/bidirectional_rnn/bw/bw/basic_rnn_cell/MatMul_2-bidirectional_rnn/bw/basic_rnn_cell/bias/read*
T0*
data_formatNHWC
p
-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_2Tanh0bidirectional_rnn/bw/bw/basic_rnn_cell/BiasAdd_2*
T0
^
4bidirectional_rnn/bw/bw/basic_rnn_cell/concat_3/axisConst*
value	B :*
dtype0
�
/bidirectional_rnn/bw/bw/basic_rnn_cell/concat_3ConcatV2unstack-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_24bidirectional_rnn/bw/bw/basic_rnn_cell/concat_3/axis*
N*

Tidx0*
T0
�
/bidirectional_rnn/bw/bw/basic_rnn_cell/MatMul_3MatMul/bidirectional_rnn/bw/bw/basic_rnn_cell/concat_3/bidirectional_rnn/bw/basic_rnn_cell/kernel/read*
T0*
transpose_a( *
transpose_b( 
�
0bidirectional_rnn/bw/bw/basic_rnn_cell/BiasAdd_3BiasAdd/bidirectional_rnn/bw/bw/basic_rnn_cell/MatMul_3-bidirectional_rnn/bw/basic_rnn_cell/bias/read*
T0*
data_formatNHWC
p
-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_3Tanh0bidirectional_rnn/bw/bw/basic_rnn_cell/BiasAdd_3*
T0
5
concat/axisConst*
value	B :*
dtype0
�
concatConcatV2+bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_3concat/axis*

Tidx0*
T0*
N
7
concat_1/axisConst*
value	B :*
dtype0
�
concat_1ConcatV2-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_1-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_2concat_1/axis*

Tidx0*
T0*
N
7
concat_2/axisConst*
value	B :*
dtype0
�
concat_2ConcatV2-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_2-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_1concat_2/axis*
T0*
N*

Tidx0
7
concat_3/axisConst*
value	B :*
dtype0
�
concat_3ConcatV2-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_3+bidirectional_rnn/bw/bw/basic_rnn_cell/Tanhconcat_3/axis*
T0*
N*

Tidx0
7
concat_4/axisConst*
dtype0*
value	B : 
g
concat_4ConcatV2concatconcat_1concat_2concat_3concat_4/axis*

Tidx0*
T0*
N
L
concat_5Identity-bidirectional_rnn/fw/fw/basic_rnn_cell/Tanh_3*
T0
L
concat_6Identity-bidirectional_rnn/bw/bw/basic_rnn_cell/Tanh_3*
T0 