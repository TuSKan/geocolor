package geocolor

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// ExtractNeutralIndices finds the true geometric neutral axis.
// Given a [resL, resA, resB] tensor of true perceptual distances from 'Black',
// it finds the (a, b) index coordinate for each lightness slice L with the minimum distance.
func ExtractNeutralIndices(ctx *graph.Graph, dampenedDistances *graph.Node) *graph.Node {
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

// OptimalNeutralAxis takes your LCH tensor of shape [ResL, ResA, ResB, Channels]
// and returns a [ResL, 2] tensor containing the optimal (L, C) pairs.
func OptimalNeutralAxis(g *graph.Graph, colorsLCH *graph.Node) *graph.Node {
	shape := colorsLCH.Shape()
	resL := shape.Dimensions[0]
	resA := shape.Dimensions[1]
	resB := shape.Dimensions[2]
	spatialSize := resA * resB

	// 1. Flatten the spatial dimensions (A and B)
	// Shape transforms from [ResL, ResA, ResB, Channels] -> [ResL, ResA * ResB, Channels]
	flattened := graph.Reshape(colorsLCH, resL, spatialSize, -1)

	// 2. Isolate the C channel (index 1)
	// Slice shape: [ResL, spatialSize, 1]
	C := graph.Slice(flattened, graph.AxisRange(), graph.AxisRange(), graph.AxisRange(1, 2))
	// Squeeze removes the trailing dimension of size 1 -> [ResL, spatialSize]
	CSqueezed := graph.Squeeze(C, -1)

	// 3. Find the index of the minimum C value for each L slice
	// ArgMin returns the indices of shape [ResL]
	minIndices := graph.ArgMin(CSqueezed, -1, dtypes.Int32)

	// 4. Create a One-Hot mask to isolate the winning voxels
	// Shape: [ResL, spatialSize]
	oneHot := graph.OneHot(minIndices, spatialSize, flattened.DType())
	// Expand to multiply against the Channels: [ResL, spatialSize, 1]
	oneHotMask := graph.ExpandAxes(oneHot, -1)

	// 5. Apply the mask and sum it up
	// Multiplying zeroes out all non-minimum voxels.
	// ReduceSum collapses the spatial dimension, leaving only the winning values!
	// Shape: [ResL, Channels]
	maskedVoxels := graph.Mul(flattened, oneHotMask)
	targetVoxels := graph.ReduceSum(maskedVoxels, 1)

	// 6. Slice out just the L (index 0) and C (index 1) channels to return
	// Final Shape: [ResL, 2]
	neutralAxisLC := graph.Slice(targetVoxels, graph.AxisRange(), graph.AxisRange(0, 2))

	return neutralAxisLC
}
