function visualize(timeseries, partition)
    if size(timeseries)[1] < 4
        fig = Figure()
        ax = LScene(fig[1,1]; show_axis=false)
        scatter!(ax, timeseries, color=partition, colormap=:glasbey_hv_n256, markersize=10)
        display(fig)
        return fig
    else 
        println("dimensions too high for visualization")
        return false 
    end
end

visualize(timeseries, partition::StateSpacePartition) = visualize(timeseries, partition.partitions)

function graph_from_PI(PI)
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    adj = spzeros(Int64, N, N)
    adj_mod = spzeros(Float64, N, N)
    for i in ProgressBar(eachindex(PI))
        ii = PI[i][1]
        jj = PI[i][2]
        modularity_value = PI[i][3]
        adj[ii, jj] += 1
        adj_mod[ii, jj] = modularity_value
    end 
    N = maximum([maximum([PI[i][1], PI[i][2]]) for i in eachindex(PI)])
    node_labels = zeros(N)
    for i in eachindex(PI)
        node_labels[PI[i][2]] = PI[i][3]
    end
    node_labels[1] = 1.0
    return node_labels, adj, adj_mod, length(PI)
end