
e
in_0Const*
dtype0*I
value@B>"0��+���d>L?��=8 9?G�>K�:�NZi?�f���y?ʓ��/?
=
	in_0/readIdentityin_0*
T0*
_class
	loc:@in_0
S
moments/mean/reduction_indicesConst*
dtype0*
valueB"       
e
moments/meanMean	in_0/readmoments/mean/reduction_indices*
T0*

Tidx0*
	keep_dims(
;
moments/StopGradientStopGradientmoments/mean*
T0
X
moments/SquaredDifferenceSquaredDifference	in_0/readmoments/StopGradient*
T0
W
"moments/variance/reduction_indicesConst*
dtype0*
valueB"       
}
moments/varianceMeanmoments/SquaredDifference"moments/variance/reduction_indices*
T0*

Tidx0*
	keep_dims(
I
moments/SqueezeSqueezemoments/mean*
T0*
squeeze_dims
 
O
moments/Squeeze_1Squeezemoments/variance*
T0*
squeeze_dims
  