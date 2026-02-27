package geocolor

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// ExtractNeutralAxis finds the true geometric neutral axis.
// Given a [resL, resA, resB] tensor of true perceptual distances from 'Black',
// it finds the (a, b) index coordinate for each lightness slice L with the minimum distance.
func ExtractNeutralAxis(ctx *graph.Graph, dampenedDistances *graph.Node) *graph.Node {
	dims := dampenedDistances.Shape().Dimensions
	resL, resA, resB := dims[0], dims[1], dims[2]

	// 1. Flatten the spatial dimensions: [resL, resA, resB] -> [resL, resA * resB]
	flattened := graph.Reshape(dampenedDistances, resL, resA*resB)

	// 2. ArgMin finds the index of the minimum distance in the flattened slice
	// Output shape: [resL]
	minIndices := graph.ArgMin(flattened, -1, dtypes.Int32)

	// Scalar resB for arithmetic decoding
	resB_Tensor := graph.Const(ctx, int32(resB))
	
	// 3. Decode the 1D index back to 2D (a, b) coordinates
	// a_index = floor(index / resB)
	aIndex := graph.Div(minIndices, resB_Tensor)
	
	// b_index = index % resB
	bIndex := graph.Rem(minIndices, resB_Tensor)

	// Stack and expand dims so we get [resL, 1] then concat to [resL, 2]
	aExpanded := graph.ExpandAxes(aIndex, -1)
	bExpanded := graph.ExpandAxes(bIndex, -1)

	return graph.Concatenate([]*graph.Node{aExpanded, bExpanded}, -1)
}
