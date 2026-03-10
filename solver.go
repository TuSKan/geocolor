package geocolor

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// SolveGlobalDistances computes the global shortest path distances from a discrete origin
// voxel using an iterative Bellman-Ford/Eikonal approach across the non-Riemannian grid.
// Evaluates entirely within the XLA graph and terminates early upon convergence or max iterations.
func SolveGlobalDistances(ctx *graph.Graph, localMetrics *graph.Node, originIndex [3]int) *graph.Node {
	dims := localMetrics.Shape().Dimensions[:3]
	resL, resA, resB := dims[0], dims[1], dims[2]

	infVal := float64(1e9)
	inf := graph.Const(ctx, infVal)
	zero := graph.Const(ctx, float64(0.0))

	// Create coordinate grids to identify the origin
	shapeInt32 := shapes.Make(dtypes.Int32, resL, resA, resB)
	iotaL := graph.Iota(ctx, shapeInt32, 0)
	iotaA := graph.Iota(ctx, shapeInt32, 1)
	iotaB := graph.Iota(ctx, shapeInt32, 2)

	origL := graph.Const(ctx, int32(originIndex[0]))
	origA := graph.Const(ctx, int32(originIndex[1]))
	origB := graph.Const(ctx, int32(originIndex[2]))

	isOrig := graph.LogicalAnd(
		graph.LogicalAnd(graph.Equal(iotaL, origL), graph.Equal(iotaA, origA)),
		graph.Equal(iotaB, origB),
	)

	// Distances tensor starts as infinity everywhere except at the origin
	initialDistances := graph.Where(isOrig, zero, graph.BroadcastToDims(inf, resL, resA, resB))

	// Squeeze channel dimension on metrics for direct broadcasting
	wL := graph.Reshape(graph.SliceAxis(localMetrics, 3, graph.AxisRange(0, 1)), resL, resA, resB)
	wa := graph.Reshape(graph.SliceAxis(localMetrics, 3, graph.AxisRange(1, 2)), resL, resA, resB)
	wb := graph.Reshape(graph.SliceAxis(localMetrics, 3, graph.AxisRange(2, 3)), resL, resA, resB)

	// Dynamic loop limits (safe max iterations is the sum of grid resolutions to cross the map)
	maxIters := graph.Const(ctx, int32(resL+resA+resB))

	initialIter := graph.Const(ctx, int32(0))
	initialChanged := graph.Const(ctx, true)

	// Shapes for the loop state: [iter, dist, hasChanged, wL, wa, wb]
	shapeIter := shapes.Make(dtypes.Int32)
	shapeBool := shapes.Make(dtypes.Bool)
	shapeGrid := shapes.Make(dtypes.Float64, resL, resA, resB)

	cond := graph.NewClosure(ctx, func(g *graph.Graph) []*graph.Node {
		iter := graph.Parameter(g, "iter", shapeIter)
		_ = graph.Parameter(g, "dist", shapeGrid)
		hasChanged := graph.Parameter(g, "hasChanged", shapeBool)
		_ = graph.Parameter(g, "wL", shapeGrid)
		_ = graph.Parameter(g, "wa", shapeGrid)
		_ = graph.Parameter(g, "wb", shapeGrid)

		// Continue if iter < maxIters AND hasChanged is true
		iterCheck := graph.LessThan(iter, maxIters)
		return []*graph.Node{graph.LogicalAnd(iterCheck, hasChanged)}
	})

	body := graph.NewClosure(ctx, func(g *graph.Graph) []*graph.Node {
		iter := graph.Parameter(g, "iter", shapeIter)
		dist := graph.Parameter(g, "dist", shapeGrid)
		_ = graph.Parameter(g, "hasChanged", shapeBool)
		wL_body := graph.Parameter(g, "wL", shapeGrid)
		wa_body := graph.Parameter(g, "wa", shapeGrid)
		wb_body := graph.Parameter(g, "wb", shapeGrid)

		infBody := graph.Const(g, infVal)

		// Helper to shift distance and weight grids to evaluate the incoming edge from a neighbor
		computeDirectionalCost := func(wAxis *graph.Node, axis int, dir int) *graph.Node {
			res := dims[axis]
			padConfig := make([]graph.PadAxis, dist.Rank())

			if dir == 1 { // Incoming from i-1
				padConfig[axis] = graph.PadAxis{Start: 1, End: 0}
				shiftedDist := graph.SliceAxis(graph.Pad(dist, infBody, padConfig...), axis, graph.AxisRange(0, res))
				shiftedWAxis := graph.SliceAxis(graph.Pad(wAxis, infBody, padConfig...), axis, graph.AxisRange(0, res))
				return graph.Add(shiftedDist, shiftedWAxis)
			} else { // Incoming from i+1
				padConfig[axis] = graph.PadAxis{Start: 0, End: 1}
				shiftedDist := graph.SliceAxis(graph.Pad(dist, infBody, padConfig...), axis, graph.AxisRange(1, res+1))
				// Going i->i+1 uses the local weights originally specified at i
				return graph.Add(shiftedDist, wAxis)
			}
		}

		// Calculate incoming cost from all 6 directions using the Pad/Slice finite difference mimicry
		c_L_minus := computeDirectionalCost(wL_body, 0, 1)
		c_L_plus := computeDirectionalCost(wL_body, 0, -1)
		c_a_minus := computeDirectionalCost(wa_body, 1, 1)
		c_a_plus := computeDirectionalCost(wa_body, 1, -1)
		c_b_minus := computeDirectionalCost(wb_body, 2, 1)
		c_b_plus := computeDirectionalCost(wb_body, 2, -1)

		// Find minimum distance resolving from all directions
		newDist := graph.Min(dist, c_L_minus)
		newDist = graph.Min(newDist, c_L_plus)
		newDist = graph.Min(newDist, c_a_minus)
		newDist = graph.Min(newDist, c_a_plus)
		newDist = graph.Min(newDist, c_b_minus)
		newDist = graph.Min(newDist, c_b_plus)

		// Detect if any cell changed meaning convergence hasn't been reached
		diff := graph.Sub(dist, newDist)
		changed := graph.GreaterThan(graph.ReduceMax(diff), graph.Const(g, float64(1e-5)))

		// Increment loop counter
		nextIter := graph.Add(iter, graph.Const(g, int32(1)))

		return []*graph.Node{nextIter, newDist, changed, wL_body, wa_body, wb_body}
	})

	results := graph.While(cond, body, initialIter, initialDistances, initialChanged, wL, wa, wb)

	// results[1] represents the converged distance tensor
	return results[1]
}
