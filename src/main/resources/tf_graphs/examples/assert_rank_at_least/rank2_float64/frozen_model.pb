
e
in_0Const*
dtype0*I
value@B>"0      @      @      @      @      @      @
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
.
in_1Const*
dtype0*
value	B :
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
O
assert_rank_at_least/ShapeConst*
dtype0*
valueB"      
C
assert_rank_at_least/RankConst*
dtype0*
value	B :
`
!assert_rank_at_least/GreaterEqualGreaterEqualassert_rank_at_least/Rank	in_1/read*
T0
H
@assert_rank_at_least/assert_rank/static_checks_determined_all_okNoOp
�
'assert_rank_at_least/control_dependencyIdentity!assert_rank_at_least/GreaterEqualA^assert_rank_at_least/assert_rank/static_checks_determined_all_ok*
T0
*4
_class*
(&loc:@assert_rank_at_least/GreaterEqual
R
)assert_rank_at_least/Assert/Assert/data_0Const*
dtype0*
valueB B 
|
)assert_rank_at_least/Assert/Assert/data_1Const*
dtype0*;
value2B0 B*Tensor in_0/read:0 must have rank at least
b
)assert_rank_at_least/Assert/Assert/data_3Const*
dtype0*!
valueB BReceived shape: 
�
"assert_rank_at_least/Assert/AssertAssert'assert_rank_at_least/control_dependency)assert_rank_at_least/Assert/Assert/data_0)assert_rank_at_least/Assert/Assert/data_1	in_1/read)assert_rank_at_least/Assert/Assert/data_3assert_rank_at_least/Shape*
T	
2*
	summarize
V
Cast_1Cast	in_1/read#^assert_rank_at_least/Assert/Assert*

DstT0*

SrcT0
&
AddAdd	in_0/readCast_1*
T0 