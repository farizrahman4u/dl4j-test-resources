
.
in_0Const*
dtype0*
value	B :
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
.
in_1Const*
dtype0*
value	B :
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
7
assert_less/LessLess	in_0/read	in_1/read*
T0
:
assert_less/ConstConst*
dtype0*
valueB 
X
assert_less/AllAllassert_less/Lessassert_less/Const*

Tidx0*
	keep_dims( 
I
 assert_less/Assert/Assert/data_0Const*
dtype0*
valueB B 
s
 assert_less/Assert/Assert/data_1Const*
dtype0*;
value2B0 B*Condition x < y did not hold element-wise:
[
 assert_less/Assert/Assert/data_2Const*
dtype0*#
valueB Bx (in_0/read:0) = 
[
 assert_less/Assert/Assert/data_4Const*
dtype0*#
valueB By (in_1/read:0) = 
�
assert_less/Assert/AssertAssertassert_less/All assert_less/Assert/Assert/data_0 assert_less/Assert/Assert/data_1 assert_less/Assert/Assert/data_2	in_0/read assert_less/Assert/Assert/data_4	in_1/read*
T

2*
	summarize
E
AddAdd	in_0/read	in_1/read^assert_less/Assert/Assert*
T0 