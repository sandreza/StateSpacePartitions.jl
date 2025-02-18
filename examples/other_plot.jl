Q = generator(ssp.partitions)

R, I = decomposition(Q)

Λ, W =  eigen(R')

W[:, end]

koopman = W[:, end-3:end-1]

scatter(koopman')