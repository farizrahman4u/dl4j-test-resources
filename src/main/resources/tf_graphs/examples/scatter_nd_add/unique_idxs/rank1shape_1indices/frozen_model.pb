
Y
in_0Const*=
value4B2
"(  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?*
dtype0
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
A
in_1Const*%
valueB"      	      *
dtype0
=
	in_1/readIdentityin_1*
T0*
_class
	loc:@in_1
B
Reshape/shapeConst*
valueB"      *
dtype0
C
ReshapeReshape	in_1/readReshape/shape*
T0*
Tshape0
A
in_2Const*%
valueB"E?m?ζ|?Ψ²χ>*
dtype0
=
	in_2/readIdentityin_2*
T0*
_class
	loc:@in_2
]
VariableConst*=
value4B2
"(  ?  ?  ?Άμ½?Θβ?  ?  ?AΛφ?  ?σώ?*
dtype0
t
AssignAssignVariable	in_0/read*
use_locking(*
T0*
_class
loc:@Variable*
validate_shape(

ScatterNdAddScatterNdAddAssignReshape	in_2/read*
use_locking( *
Tindices0*
T0*
_class
loc:@Variable 