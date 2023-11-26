function lorenz(s)
    x, y, z = s
    ẋ = 10.0 * (y - x)
    ẏ = x * (28.0 - z) - y
    ż = x * y - (8 / 3) * z
    return [ẋ, ẏ, ż]
end

function rossler(s; a = 0.2, b = 0.2, c = 5.7)
    x, y, z = s
    ẋ = -y - z
    ẏ = x + a * y
    ż = b + z * (x - c)
    return [ẋ, ẏ, ż]
end

function aizawa(s; a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.25)
    x, y, z = s
    ẋ = (z - b) * x - d * y
    ẏ = d * x + (z - b) * y
    ż = c + a * z - z^3/3 - (x^2 + y^2) * (1 + e * z) + f * z * x^3
    return [ẋ, ẏ, ż]
end