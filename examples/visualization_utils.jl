using MarkovChainHammer, LinearAlgebra

function visualize(trajectory, partition)
    if size(trajectory)[1] < 4
        fig = Figure()
        ax = LScene(fig[1,1]; show_axis=false)
        scatter!(ax, trajectory, color=partition, colormap=:glasbey_hv_n256, markersize=10)
        display(fig)
        return fig
    else 
        println("dimensions too high for visualization")
        return false 
    end
end

visualize(trajectory, partition::StateSpacePartition) = visualize(trajectory, partition.partitions)

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

function visualize_koopman_mode(trajectory, partitions; mode = 2, colormap1 = :balance, colormap2 = :cyclic_mrybm_35_75_c68_n256, markersize = 10)
    if (size(trajectory)[1] < 4) & (maximum(partitions) < 4000)
        generator_matrix = generator(partitions)
        eigenvalues, eigenvectors = eigen(generator_matrix')
        λ = reverse(eigenvalues)[mode]
        v = reverse(eigenvectors, dims = 2)[:, mode]
        if ((imag(λ) / abs(λ)) < eps(100.0)) | (abs(λ) < eps(100.0))
            set_theme!(backgroundcolor = :black)
            w = real.(v)
            koopman = [w[partition[i]] for i in eachindex(partition)]
            fig = Figure()
            ax = LScene(fig[1,1]; show_axis=false)
            scatter!(ax, trajectory, color=koopman, colormap= colormap1, markersize=markersize)
            display(fig)
            return fig
        else
            set_theme!(backgroundcolor = :black)
            koopman = [v[partitions[i]] for i in eachindex(partitions)]
            koopman_amp = abs.(koopman)
            koopman_phase = angle.(koopman)
            fig = Figure()
            ax = LScene(fig[1,1]; show_axis=false)
            scatter!(ax, trajectory, color=koopman_amp, colormap = colormap1, markersize=markersize)
            ax = LScene(fig[1,2]; show_axis=false)
            scatter!(ax, trajectory, color=koopman_phase, colormap= colormap2, markersize=markersize)
            display(fig)
            return fig
        end
    else 
        println("dimensions too high for visualization")
        return false 
    end
end