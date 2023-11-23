using StateSpacePartitions, ProgressBars

function lorenz!(ṡ, s)
    ṡ[1] = 10.0 * (s[2] - s[1])
    ṡ[2] = s[1] * (28.0 - s[3]) - s[2]
    ṡ[3] = s[1] * s[2] - (8 / 3) * s[3]
    return nothing
end

function rk4(f, s, dt)
    ls = length(s)
    k1 = zeros(ls)
    k2 = zeros(ls)
    k3 = zeros(ls)
    k4 = zeros(ls)
    f(k1, s)
    f(k2, s + k1 * dt / 2)
    f(k3, s + k2 * dt / 2)
    f(k4, s + k3 * dt)
    return s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
end

dt = 0.1 
iterations = 1000 

timeseries = zeros(3, iterations)
timeseries[:, 1] .= [14.0, 20.0, 27.0]
for i in ProgressBar(2:iterations)
    state = rk4(lorenz!, timeseries[:, i-1], dt)
    timeseries[:, i] .= state
end
state_space_partitions = StateSpacePartition(timeseries)
