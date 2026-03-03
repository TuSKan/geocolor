package geocolor

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

type GridConfig struct {
	ResL, ResA, ResB int        // Number of voxels in L, a, b dimensions
	MinBounds        [3]float64 // Minimum continuous (L, a, b) coordinates
	MaxBounds        [3]float64 // Maximum continuous (L, a, b) coordinates
	OriginIndex      [3]int     // Discrete voxel grid index representing the "Black" origin
}

// BuildCoordinateGridGraph generates a dense 3D grid of float32 coordinates
// representing L, a, and b channels using GoMLX native graphing operations.
// Returns a stacked tensor of shape [resL, resA, resB, 3].
func BuildCoordinateGridGraph(ctx *graph.Graph, cfg *GridConfig) *graph.Node {
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
		return graph.BroadcastToDims(val, cfg.ResL, cfg.ResA, cfg.ResB, 1)
	}

	valL := scaleAxis(cfg.ResL, cfg.MinBounds[0], cfg.MaxBounds[0], 0)
	valA := scaleAxis(cfg.ResA, cfg.MinBounds[1], cfg.MaxBounds[1], 1)
	valB := scaleAxis(cfg.ResB, cfg.MinBounds[2], cfg.MaxBounds[2], 2)

	// Combine all three scalar channels into a single depth-3 tensor
	return graph.Concatenate([]*graph.Node{valL, valA, valB}, 3)
}
