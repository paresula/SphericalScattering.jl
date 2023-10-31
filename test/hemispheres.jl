sp = HemispherePEC(1.0,Medium(1.0,1.0),Medium(1.0,1.0))
exc = UniformField(embedding = Medium(1.0,1.0), amplitude=1.0,direction=SVector{3,Float64}(0,0,1))
point = SVector{3,Float64}(0.0,0,-0.5)

xs = range(-2.0, step = 0.05, stop = 2.0)
zs = range(-2.0, step = 0.05, stop = 2.0)

points = [SVector{3,Float64}(x,0,z) for x in xs for z in zs]

#scatteredfield(sp, exc, ScalarPotential(points))
pot = scatteredfield(sp, exc, points, ScalarPotential(points);N=1600)
pot2 = reshape(pot, length(xs) ,length(zs))

Plotly.plot([contour(z = pot2, ncontours = 40)])