package geocolor

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
)

// ComputeGeometricColor uses the calculated geometric neutral indices to shift the
// original (L, a, b) continuous geometry so the origin lies exactly on the neutral axis.
// It then calculates Geometric Chroma and Geometric Hue based on this shifted space.
func ComputeGeometricColor(ctx *graph.Graph, coords *graph.Node, neutralIndices *graph.Node, minBounds, maxBounds [3]float64) *graph.Node {
	dims := coords.Shape().Dimensions
	resL, resA, resB := dims[0], dims[1], dims[2]

	aMin, aMax := minBounds[1], maxBounds[1]
	bMin, bMax := minBounds[2], maxBounds[2]

	// 1. Convert integer neutral indices [resL, 2] to continuous coordinates [resL, 2]
	// First split indices into a_idx and b_idx
	aIdx := graph.SliceAxis(neutralIndices, 1, graph.AxisRange(0, 1))
	bIdx := graph.SliceAxis(neutralIndices, 1, graph.AxisRange(1, 2))

	// Cast to Float32 for math
	aIdxF := graph.ConvertType(aIdx, dtypes.Float32)
	bIdxF := graph.ConvertType(bIdx, dtypes.Float32)

	// Scale integer index to continuous space: val = idx * scale + min
	// where scale = (max - min) / (res - 1)
	resAMinus1 := float32(resA - 1)
	if resAMinus1 <= 0 { resAMinus1 = 1.0 }
	scaleA := graph.Const(ctx, float32(aMax - aMin) / resAMinus1)
	minA := graph.Const(ctx, float32(aMin))
	aNeutral := graph.Add(graph.Mul(aIdxF, scaleA), minA) // [resL, 1]

	resBMinus1 := float32(resB - 1)
	if resBMinus1 <= 0 { resBMinus1 = 1.0 }
	scaleB := graph.Const(ctx, float32(bMax - bMin) / resBMinus1)
	minB := graph.Const(ctx, float32(bMin))
	bNeutral := graph.Add(graph.Mul(bIdxF, scaleB), minB) // [resL, 1]

	// 2. Broadcast neutral coordinates to full grid shape
	// Target shape for subtraction is [resL, resA, resB]
	
	// Reshape to [resL, 1, 1] so it broadcasts across A and B axes
	aNeutral3D := graph.Reshape(aNeutral, resL, 1, 1)
	bNeutral3D := graph.Reshape(bNeutral, resL, 1, 1)

	// Broadcast
	aNeutralGrid := graph.BroadcastToDims(aNeutral3D, resL, resA, resB)
	bNeutralGrid := graph.BroadcastToDims(bNeutral3D, resL, resA, resB)

	// 3. Extract original components
	L_orig := graph.SliceAxis(coords, 3, graph.AxisRange(0, 1))
	a_orig := graph.SliceAxis(coords, 3, graph.AxisRange(1, 2))
	b_orig := graph.SliceAxis(coords, 3, graph.AxisRange(2, 3))

	// Reshape to remove trailing 1 dimension so shapes match perfectly
	L_grid := graph.Reshape(L_orig, resL, resA, resB)
	a_grid := graph.Reshape(a_orig, resL, resA, resB)
	b_grid := graph.Reshape(b_orig, resL, resA, resB)

	// 4. Shift Grid
	a_shifted := graph.Sub(a_grid, aNeutralGrid)
	b_shifted := graph.Sub(b_grid, bNeutralGrid)

	// 5. Calculate Geometric Chroma: C_geo = Sqrt(a_shifted^2 + b_shifted^2)
	a_shifted_sq := graph.Square(a_shifted)
	b_shifted_sq := graph.Square(b_shifted)
	C_geo := graph.Sqrt(graph.Add(a_shifted_sq, b_shifted_sq))

	// 6. Calculate Geometric Hue: h_geo = Atan2(b_shifted, a_shifted)
	h_geo := graph.Atan2(b_shifted, a_shifted)

	// Expand dims to stack them
	L_grid_expanded := graph.ExpandAxes(L_grid, -1)
	C_geo_expanded := graph.ExpandAxes(C_geo, -1)
	h_geo_expanded := graph.ExpandAxes(h_geo, -1)

	// 7. Concatenate into [resL, resA, resB, 3]
	return graph.Concatenate([]*graph.Node{L_grid_expanded, C_geo_expanded, h_geo_expanded}, -1)
}
