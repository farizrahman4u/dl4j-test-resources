
}
in_0Const*
dtype0*a
valueXBV"H  �@   A  �@   @  �@  �?  �@   A  �@  A      �@  �@  �@  �?  @@  �?  �@
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
I
in_1Const*
dtype0*-
value$B""                   
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
G
UnsortedSegmentMean/ShapeConst*
dtype0*
valueB:
K
UnsortedSegmentMean/ones/ConstConst*
dtype0*
valueB
 *  �?
v
UnsortedSegmentMean/onesFillUnsortedSegmentMean/ShapeUnsortedSegmentMean/ones/Const*
T0*

index_type0
]
3UnsortedSegmentMean/UnsortedSegmentSum/num_segmentsConst*
dtype0*
value	B :
�
&UnsortedSegmentMean/UnsortedSegmentSumUnsortedSegmentSumUnsortedSegmentMean/ones	in_1/read3UnsortedSegmentMean/UnsortedSegmentSum/num_segments*
T0*
Tindices0*
Tnumsegments0
V
!UnsortedSegmentMean/Reshape/shapeConst*
dtype0*
valueB"      
�
UnsortedSegmentMean/ReshapeReshape&UnsortedSegmentMean/UnsortedSegmentSum!UnsortedSegmentMean/Reshape/shape*
T0*
Tshape0
J
UnsortedSegmentMean/Maximum/yConst*
dtype0*
valueB
 *  �?
k
UnsortedSegmentMean/MaximumMaximumUnsortedSegmentMean/ReshapeUnsortedSegmentMean/Maximum/y*
T0
_
5UnsortedSegmentMean/UnsortedSegmentSum_1/num_segmentsConst*
dtype0*
value	B :
�
(UnsortedSegmentMean/UnsortedSegmentSum_1UnsortedSegmentSum	in_0/read	in_1/read5UnsortedSegmentMean/UnsortedSegmentSum_1/num_segments*
T0*
Tindices0*
Tnumsegments0
v
UnsortedSegmentMean/truedivRealDiv(UnsortedSegmentMean/UnsortedSegmentSum_1UnsortedSegmentMean/Maximum*
T0 