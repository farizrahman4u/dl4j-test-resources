
�
in_0Const*
dtype0*i
value`B^"P~^G?LM?�p9?�ol>�%:?X�8><q?b|d?��?�al?P@�=,5K?ֹ(?�6?`u#>0�>�{>�h�>�o~?v|?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
q
in_1Const*
dtype0*U
valueLBJ"<�E?��m?�|?ز�>��$?@�?�n&?��B?ܰB?��>ps?�*\?`�I?��d?w�>
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
q
in_2Const*
dtype0*U
valueLBJ"<��q?�~?�]?�O�>�?|;:?�|�>�}H?�H>�<?`�|=��)? �=D˦>��?
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
A
in_3Const*
dtype0*%
valueB"             
=
	in_3/readIdentityin_3*
T0*
_class
	loc:@in_3
U
embedding_lookup/Reshape/shapeConst*
dtype0*
valueB:
���������
e
embedding_lookup/ReshapeReshape	in_3/readembedding_lookup/Reshape/shape*
T0*
Tshape0
?
embedding_lookup/SizeConst*
dtype0*
value	B :
F
embedding_lookup/range/startConst*
dtype0*
value	B : 
F
embedding_lookup/range/deltaConst*
dtype0*
value	B :
~
embedding_lookup/rangeRangeembedding_lookup/range/startembedding_lookup/Sizeembedding_lookup/range/delta*

Tidx0
@
embedding_lookup/mod/yConst*
dtype0*
value	B :
[
embedding_lookup/modFloorModembedding_lookup/Reshapeembedding_lookup/mod/y*
T0
E
embedding_lookup/floordiv/yConst*
dtype0*
value	B :
e
embedding_lookup/floordivFloorDivembedding_lookup/Reshapeembedding_lookup/floordiv/y*
T0
�
!embedding_lookup/DynamicPartitionDynamicPartitionembedding_lookup/floordivembedding_lookup/mod*
T0*
num_partitions
�
#embedding_lookup/DynamicPartition_1DynamicPartitionembedding_lookup/rangeembedding_lookup/mod*
T0*
num_partitions
a
embedding_lookup/GatherV2/axisConst*
_class
	loc:@in_0*
dtype0*
value	B : 
�
embedding_lookup/GatherV2GatherV2	in_0/read!embedding_lookup/DynamicPartitionembedding_lookup/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0*
_class
	loc:@in_0
c
 embedding_lookup/GatherV2_1/axisConst*
_class
	loc:@in_1*
dtype0*
value	B : 
�
embedding_lookup/GatherV2_1GatherV2	in_1/read#embedding_lookup/DynamicPartition:1 embedding_lookup/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_class
	loc:@in_1
c
 embedding_lookup/GatherV2_2/axisConst*
_class
	loc:@in_2*
dtype0*
value	B : 
�
embedding_lookup/GatherV2_2GatherV2	in_2/read#embedding_lookup/DynamicPartition:2 embedding_lookup/GatherV2_2/axis*
Taxis0*
Tindices0*
Tparams0*
_class
	loc:@in_2
�
embedding_lookupParallelDynamicStitch#embedding_lookup/DynamicPartition_1%embedding_lookup/DynamicPartition_1:1%embedding_lookup/DynamicPartition_1:2embedding_lookup/GatherV2embedding_lookup/GatherV2_1embedding_lookup/GatherV2_2*
N*
T0
D
embedding_lookup/ShapeConst*
dtype0*
valueB:
N
 embedding_lookup/concat/values_1Const*
dtype0*
valueB:
F
embedding_lookup/concat/axisConst*
dtype0*
value	B : 
�
embedding_lookup/concatConcatV2embedding_lookup/Shape embedding_lookup/concat/values_1embedding_lookup/concat/axis*
N*
T0*

Tidx0
g
embedding_lookup/Reshape_1Reshapeembedding_lookupembedding_lookup/concat*
T0*
Tshape0
i
!embedding_lookup/clip_by_norm/mulMulembedding_lookup/Reshape_1embedding_lookup/Reshape_1*
T0
a
3embedding_lookup/clip_by_norm/Sum/reduction_indicesConst*
dtype0*
valueB:
�
!embedding_lookup/clip_by_norm/SumSum!embedding_lookup/clip_by_norm/mul3embedding_lookup/clip_by_norm/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims(
V
"embedding_lookup/clip_by_norm/SqrtSqrt!embedding_lookup/clip_by_norm/Sum*
T0
R
%embedding_lookup/clip_by_norm/mul_1/yConst*
dtype0*
valueB
 *  �?
v
#embedding_lookup/clip_by_norm/mul_1Mulembedding_lookup/Reshape_1%embedding_lookup/clip_by_norm/mul_1/y*
T0
T
'embedding_lookup/clip_by_norm/Maximum/yConst*
dtype0*
valueB
 *  �?
�
%embedding_lookup/clip_by_norm/MaximumMaximum"embedding_lookup/clip_by_norm/Sqrt'embedding_lookup/clip_by_norm/Maximum/y*
T0
�
%embedding_lookup/clip_by_norm/truedivRealDiv#embedding_lookup/clip_by_norm/mul_1%embedding_lookup/clip_by_norm/Maximum*
T0
Y
embedding_lookup/clip_by_normIdentity%embedding_lookup/clip_by_norm/truediv*
T0 