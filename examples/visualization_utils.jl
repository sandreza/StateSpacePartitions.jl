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