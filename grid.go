package geocolor

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// BuildCoordinateGridGraph generates a dense 3D grid of float32 coordinates
// representing L, a, and b channels using GoMLX native graphing operations.
// Returns a stacked tensor of shape [resL, resA, resB, 3].
func BuildCoordinateGridGraph(ctx *graph.Graph, resL, resA, resB int, minBounds, maxBounds [3]float64) *graph.Node {
	// Helper function to create an axis grid using Iota and scale it.
	scaleAxis := func(res int, minBound, maxBound float64, axis int) *graph.Node {
		// Initialize shape dimension where the current axis has size 'res' 
		// and all other axes up to depth 4 have size 1.
		dims := []int{1, 1, 1, 1}
		dims[axis] = res

		// Generate index scalar sequence 0, 1, ..., res-1 along the axis
		iotaNode := graph.Iota(ctx, shapes.Make(dtypes.Float32, dims...), axis)

		resMinus1 := float64(res - 1)
		if resMinus1 <= 0 {
			resMinus1 = 1.0 // Safe fallback
		}

		// Compute the static scale multiplier and min bound constants
		scaleVal := (maxBound - minBound) / resMinus1
		scaleNode := graph.Scalar(ctx, dtypes.Float32, scaleVal)
		minNode := graph.Scalar(ctx, dtypes.Float32, minBound)

		// Equation: coord = iota * scale + min
		val := graph.Add(graph.Mul(iotaNode, scaleNode), minNode)
		
		// Broadcast the dimension to span the full [resL, resA, resB, 1] grid
		return graph.BroadcastToDims(val, resL, resA, resB, 1)
	}

	valL := scaleAxis(resL, minBounds[0], maxBounds[0], 0)
	valA := scaleAxis(resA, minBounds[1], maxBounds[1], 1)
	valB := scaleAxis(resB, minBounds[2], maxBounds[2], 2)

	// Combine all three scalar channels into a single depth-3 tensor
	return graph.Concatenate([]*graph.Node{valL, valA, valB}, 3)
}
