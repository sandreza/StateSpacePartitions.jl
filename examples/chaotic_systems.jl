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

function newton_leipnick(s; a = 5, b = 0.5, c = 10.0, d = 0.175)
    x, y, z = s
    ẋ = -b * x + y + c * y * z
    ẏ = -x - b * y + a * z * x
    ż = d * z - a * x * y
    return [ẋ, ẏ, ż]
end

function sprott_case_m(s; a = 1.7, b = 1.7)
    x, y, z = s
    ẋ = -z
    ẏ = -x^2 - y
    ż = a + b * x + y
    return [ẋ, ẏ, ż]
end

function sprott_case_a(s)
    x, y, z = s
    ẋ = y
    ẏ = -x + y * z
    ż = 1 - y^2
    return [ẋ, ẏ, ż]
end

function jd_5(s)
    x, y, z = s
    ẋ = y
    ẏ = z
    ż = -ẋ + 3 * y^2 - x^2 - x * z
    return [ẋ, ẏ, ż]
end

halvorsen_base(x, y, z; a = 1.3) = -a * x - 4 * y - 4 * z - y^2
function halvorsen(s; a = 1.4)
    x, y, z = s
    ẋ = halvorsen_base(x, y, z; a)
    ẏ = halvorsen_base(y, z, x; a)
    ż = halvorsen_base(z, x, y; a)
    return [ẋ, ẏ, ż]
end

piecewise_base(x, y, z) = 1 - x - y - 4 * abs(y)
function piecewise(s)
    x, y, z = s
    ẋ = piecewise_base(x, y, z)
    ẏ = piecewise_base(y, z, x)
    ż = piecewise_base(z, x, y)
    return [ẋ, ẏ, ż]
end