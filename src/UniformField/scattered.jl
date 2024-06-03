
"""
    scatteredfield(sphere::Sphere, excitation::PlaneWave, quantity::Field; parameter::Parameter=Parameter())

Compute the electric field scattered by a sphere, for an incident uniform field.
"""
function scatteredfield(sphere::Sphere, excitation::UniformField, quantity::Field; parameter::Parameter=Parameter())

    F = zeros(fieldType(quantity), size(quantity.locations))

    # --- compute field in Cartesian representation
    for (ind, point) in enumerate(quantity.locations)
        F[ind] = scatteredfield(sphere, excitation, point, quantity; parameter=parameter)
    end

    return F
end



"""
    scatteredfield(sphere::DielectricSphere, excitation::UniformField, point, quantity::ElectricField; parameter::Parameter=Parameter())

Compute the electric field scattered by a Dielectric sphere, for an incident uniform field with polarization in given direction.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::DielectricSphere, excitation::UniformField, point, quantity::ElectricField; parameter::Parameter=Parameter()
)

    ε0 = excitation.embedding.ε
    ε1 = sphere.filling.ε
    E0 = field(excitation, point, quantity)

    R = sphere.radius
    r = norm(point)

    if r <= R # inside the sphere 
        return (3 * ε0 / (ε1 + 2 * ε0) - 1) * E0 # Sihvola&Lindell 1988, gradient of (9) minus the incident part
    end

    ratio = (ε1 - ε0) / (ε1 + 2 * ε0) * R^3

    return E0 * (-ratio / r^3) + 3 * ratio * point / r^5 * dot(E0, point) # Sihvola&Lindell 1988, gradient of (8) minus incident part
end



"""
    scatteredfield(sphere::DielectricSphere, excitation::UniformField, point, quantity::ScalarPotential; parameter::Parameter=Parameter())

Compute the scalar potential scattered by a dielectric sphere, for an incident uniform field with polarization in given direction.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::DielectricSphere, excitation::UniformField, point, quantity::ScalarPotential; parameter::Parameter=Parameter()
)

    ε0 = excitation.embedding.ε
    ε1 = sphere.filling.ε
    Φ0 = field(excitation, point, quantity)

    R = sphere.radius
    r = norm(point)

    if r <= R # inside the sphere 
        return (3 * ε0 / (ε1 + 2 * ε0) - 1) * Φ0 # Sihvola&Lindell 1988, (9) minus the incident part
    end

    # outside the sphere
    return -Φ0 * ((ε1 - ε0) / (ε1 + 2 * ε0) * R^3 / r^3) # Sihvola&Lindell 1988, (8) minus the incident part
end



"""
    scatteredfield(sphere::DielectricSphereThinImpedanceLayer, excitation::UniformField, point, quantity::ElectricField; parameter::Parameter=Parameter())

Compute the electric field scattered by a dielectric sphere with a thin coating, where the displacement field in the coating is only in radial direction.
We assume an an incident uniform field with polarization in the given direction.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::DielectricSphereThinImpedanceLayer,
    excitation::UniformField,
    point,
    quantity::ElectricField;
    parameter::Parameter=Parameter(),
)

    point = rotate(excitation, [point]; inverse=true)[1]
    point_sph = cart2sph(point)

    r = norm(point)

    ẑ = SVector(0.0, 0.0, 1.0) #excitation.direction
    cosθ = dot(ẑ, point) / r
    sinθ = norm(cross(ẑ, point)) / r

    E₀ = excitation.amplitude

    A, K = scatterCoeff(sphere, excitation)

    if r > sphere.radius
        E = -SVector((-2 * A / r^3) * cosθ, (+A / r^3) * (-sinθ), 0.0)
    else
        E = -SVector(((E₀ - K) * cosθ, (-E₀ + K) * sinθ, 0.0))
    end

    E_cart = convertSpherical2Cartesian(E, point_sph)

    return rotate(excitation, [E_cart]; inverse=false)[1]
end



"""
    scatteredfield(sphere::DielectricSphereThinImpedanceLayer, excitation::UniformField, point, quantity::DisplacementField; parameter::Parameter=Parameter())

Compute the displacement field D = ε * E.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::DielectricSphereThinImpedanceLayer,
    excitation::UniformField,
    point,
    quantity::DisplacementField;
    parameter::Parameter=Parameter(),
)

    E = scatteredfield(sphere, excitation, point, ElectricField(quantity.locations); parameter=parameter)

    if norm(point) > sphere.radius
        D = excitation.embedding.ε * E
    else
        D = sphere.filling.ε * E
    end

    return D
end



"""
    scatteredfield(sphere::DielectricSphereThinImpedanceLayer, excitation::UniformField, point, quantity::ScalarPotentialJump; parameter::Parameter=Parameter())

Compute the jump of the scalar potential for a dielectric sphere with a thin coating, where the displacement field in the coating is only in radial direction.
We assume an an incident uniform field with polarization in the given direction.

More precisely, we compute the difference Δ = Φ_i - Φ_e, where Φ_i is the potential on the inner side and ϕ_e the exterior potential.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::DielectricSphereThinImpedanceLayer,
    excitation::UniformField,
    point,
    quantity::ScalarPotentialJump;
    parameter::Parameter=Parameter(),
)

    cosθ = dot(excitation.direction, point) / norm(point)

    ~, K = scatterCoeff(sphere, excitation)

    return sphere.thickness * (sphere.filling.ε / sphere.thinlayer.ε) * K * cosθ

end



"""
    scatteredfield(sphere::DielectricSphereThinImpedanceLayer, excitation::UniformField, point, quantity::ScalarPotential; parameter::Parameter=Parameter())

Compute the scalar potential scattered by a dielectric sphere with a thin coating, where the displacement field in the coating is only in radial direction.
We assume an an incident uniform field with polarization in the given direction.

The point and the returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::DielectricSphereThinImpedanceLayer,
    excitation::UniformField,
    point,
    quantity::ScalarPotential;
    parameter::Parameter=Parameter(),
)

    cosθ = dot(excitation.direction, point) / norm(point)
    r = norm(point)

    R = sphere.radius

    E₀ = excitation.amplitude
    A, K = scatterCoeff(sphere, excitation)

    if r >= R
        return (+A / r^2) * cosθ # Jones 1995, (C.1a)
    else
        return ((E₀ - K) * r * cosθ)
    end
end



"""
    scatterCoeff(sp::DielectricSphereThinImpedanceLayer, ex::UniformField)

Compute the expansion coefficients for the thin impedance layer case and a uniform static field excitation.
"""
function scatterCoeff(sp::DielectricSphereThinImpedanceLayer, ex::UniformField)
    R = sp.radius
    Δ = sp.thickness
    εₘ = sp.thinlayer.ε
    εₑ = ex.embedding.ε
    εᵢ = sp.filling.ε
    E₀ = ex.amplitude

    A = E₀ * R^3 * (-R * εₑ * εₘ + R * εᵢ * εₘ - Δ * εₑ * εᵢ) / (2R * εₑ * εₘ + R * εᵢ * εₘ + 2Δ * εₑ * εᵢ)
    K = 3E₀ * R * εₑ * εₘ / (2 * R * εₘ * εₑ + R * εᵢ * εₘ + 2Δ * εₑ * εᵢ)

    return A, K
end



"""
    scatteredfield(sphere::PECSphere, excitation::UniformField, point, quantity::ElectricField; parameter::Parameter=Parameter())

Compute the electric field scattered by a PEC sphere, for an incident uniform field with polarization in the given direction.

The point and returned field are in Cartesian coordinates.
"""
function scatteredfield(sphere::PECSphere, excitation::UniformField, point, quantity::ElectricField; parameter::Parameter=Parameter())

    E0 = field(excitation, point, quantity)

    R = sphere.radius
    r = norm(point)

    T = eltype(point)
    if r <= R
        return -E0
    end

    return R^3 * (-E0 / r^3 + 3 / r^5 * point * dot(E0, point)) # Griffits, Example 3.8
end



"""
    scatteredfield(sphere::PECSphere, excitation::UniformField, point, quantity::ScalarPotential; parameter::Parameter=Parameter())

Compute the scalar potential scattered by a PEC sphere, for an incident uniform field with polarization in the given direction.

The point and returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::PECSphere, excitation::UniformField, point, quantity::ScalarPotential; parameter::Parameter=Parameter()
)

    Φ0 = field(excitation, point, quantity)

    R = sphere.radius
    r = norm(point)

    T = eltype(point)
    if r <= R
        return -Φ0
    end

    return Φ0 * (-R^3 / r^3) # Griffits, Example 3.8
end



"""
    scatteredfield(sphere::LayeredSphere, excitation::UniformField, point, quantity::ElectricField; parameter::Parameter=Parameter())

Compute the electric field scattered by a layered dielectric sphere, for an incident uniform field with polarization in the given direction
using `Sihvola and Lindell, 1988, Transmission line analogy for calculating the effective permittivity of mixtures with spherical multilayer scatterers`.

In contrast to `Sihvola and Lindell` the radii of the shells are ordered from inside to outside as depicted in the documentation.

The point and returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::LayeredSphere{LN,LR,LC},
    excitation::UniformField{FC,FT,FR},
    point,
    quantity::ElectricField;
    parameter::Parameter=Parameter(),
) where {LN,LR,LC,FC,FT,FR}

    E0 = excitation.amplitude
    dir = excitation.direction
    r = norm(point)

    C, D = scatterCoeff(sphere, excitation, r)

    return E0 * (C * dir - D * dir / r^3 + 3 * D * point * dot(dir, point) / r^5 - dir)
end



"""
    scatteredfield(sphere::LayeredSphere, excitation::UniformField, point, quantity::ScalarPotential; parameter::Parameter=Parameter())

Compute the scalar potential scattered by a layered dielectric sphere, for an incident uniform field with polarization in the given direction
using `Sihvola and Lindell, 1988, Transmission line analogy for calculating the effective permittivity of mixtures with spherical multilayer scatterers`.

In contrast to `Sihvola and Lindell` the radii of the shells are ordered from inside to outside as depicted in the documentation.

The point and returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::LayeredSphere{LN,LR,LC},
    excitation::UniformField{FC,FT,FR},
    point,
    quantity::ScalarPotential;
    parameter::Parameter=Parameter(),
) where {LN,LR,LC,FC,FT,FR}

    Φ0 = field(excitation, point, quantity)
    r = norm(point)

    C, D = scatterCoeff(sphere, excitation, r)

    return (C - D / r^3 - 1.0) * Φ0
end



"""
    scatterCoeff(sphere::LayeredSphere{LN,LR,LC}, excitation::UniformField{FC,FT,FR}, r) where {LN,LR,LC,FC,FT,FR}

Scatter coefficients according to `Sihvola and Lindell`. 
However, the radii of the shells are ordered from inside to outside as depicted in the documentation.
"""
function scatterCoeff(sphere::LayeredSphere{LN,LR,LC}, excitation::UniformField{FC,FT,FR}, r) where {LN,LR,LC,FC,FT,FR}

    a = reverse(sphere.radii)
    n = length(a)
    perms = reverse(getfield.(vcat(sphere.filling, excitation.embedding), 1))

    T = promote_type(LR, LC, FC, FT, FR)

    Bk = zeros(SMatrix{2,2,T}, n)
    B = Matrix(I, 2, 2)

    for k in range(n; stop=1, step=-1)

        B11 = perms[k + 1] + 2 * perms[k]
        B12 = 2 * (perms[k + 1] - perms[k]) * a[k]^-3
        B21 = (perms[k + 1] - perms[k]) * a[k]^3
        B22 = 2 * perms[k + 1] + perms[k]

        Bk[k] = 1 / (3 * perms[k]) * ([B11 B12; B21 B22])
        B = Bk[k] * B
    end

    Ck = zeros(T, n + 1)
    Dk = zeros(T, n + 1)

    Ck[1] = 1
    Dk[n + 1] = 0
    Dk[1] = B[2, 1] / B[1, 1]
    Ck[n + 1] = 1 / B[1, 1]

    for k in range(n; stop=2, step=-1)

        Ck[k], Dk[k] = Bk[k] * [Ck[k + 1], Dk[k + 1]]
    end

    # find out in what layer `point` is
    pos = 0

    for k in 1:n
        r < a[k] && (pos = k)
    end

    C = Ck[pos + 1]
    D = Dk[pos + 1]

    return C, D
end



"""
    scatteredfield(sphere::LayeredSpherePEC, excitation::UniformField{FC,FT,FR}, point, quantity::ScalarPotential; parameter::Parameter=Parameter())

Compute the scalar potential scattered by a layered dielectric sphere with PEC core, for an incident uniform field with polarization in the given direction
using `Sihvola and Lindell, 1988, Transmission line analogy for calculating the effective permittivity of mixtures with spherical multilayer scatterers`

In contrast to `Sihvola and Lindell` the radii of the shells are ordered from inside to outside as depicted in the documentation.

The point and returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::LayeredSpherePEC{LN,LD,LR,LC},
    excitation::UniformField{FC,FT,FR},
    point,
    quantity::ScalarPotential;
    parameter::Parameter=Parameter(),
) where {LN,LD,LR,LC,FC,FT,FR}

    Φ0 = field(excitation, point, quantity)
    r = norm(point)

    C, D = scatterCoeff(sphere, excitation, r)

    return Φ0 * (C - D / r^3) - Φ0
end


"""
    scatteredfield(sphere::LayeredSpherePEC, excitation::UniformField, point, quantity::ElectricField; parameter::Parameter=Parameter())

Compute the electric field scattered by a layered dielectric sphere with PEC core, for an incident uniform field with polarization in the given direction
using `Sihvola and Lindell, 1988, Transmission line analogy for calculating the effective permittivity of mixtures with spherical multilayer scatterers`.

In contrast to `Sihvola and Lindell` the radii of the shells are ordered from inside to outside as depicted in the documentation.

The point and returned field are in Cartesian coordinates.
"""
function scatteredfield(
    sphere::LayeredSpherePEC{LN,LD,LR,LC},
    excitation::UniformField{FC,FT,FR},
    point,
    quantity::ElectricField;
    parameter::Parameter=Parameter(),
) where {LN,LD,LR,LC,FC,FT,FR}

    E0 = excitation.amplitude
    r = norm(point)
    dir = excitation.direction

    C, D = scatterCoeff(sphere, excitation, r)

    return E0 * (C * dir - D * dir / r^3 + 3 * D * point * dot(dir, point) / r^5 - dir)
end



"""
    scatterCoeff(sphere::LayeredSpherePEC{LN,LD,LR,LC}, excitation::UniformField{FC,FT,FR}, r) where {LN,LD,LR,LC,FC,FT,FR}

Scatter coefficients according to `Sihvola and Lindell`. 
However, the radii of the shells are ordered from inside to outside as depicted in the documentation.
"""
function scatterCoeff(sphere::LayeredSpherePEC{LN,LD,LR,LC}, excitation::UniformField{FC,FT,FR}, r) where {LN,LD,LR,LC,FC,FT,FR}

    a = reverse(sphere.radii)
    n = length(a) - 1
    perms = reverse(getfield.(vcat(sphere.filling, excitation.embedding), 1))

    T = promote_type(LR, LC, FC, FT, FR)

    Bk = zeros(SMatrix{2,2,T}, n)
    B = Matrix(I, 2, 2)

    for k in range(n; stop=1, step=-1)
        B11 = perms[k + 1] + 2 * perms[k]
        B12 = 2 * (perms[k + 1] - perms[k]) * a[k]^-3
        B21 = (perms[k + 1] - perms[k]) * a[k]^3
        B22 = 2 * perms[k + 1] + perms[k]

        Bk[k] = 1 / (3 * perms[k]) * ([B11 B12; B21 B22])
        B = Bk[k] * B
    end
    Ck = zeros(T, n + 1)
    Dk = zeros(T, n + 1)

    Ck[1] = 1
    Dk[1] = (B[2, 1] + B[2, 2] * a[end]^3) / (B[1, 1] + B[1, 2] * a[end]^3)
    Ck[n + 1] = 1 / (B[1, 1] + B[1, 2] * a[end]^3)
    Dk[n + 1] = a[end]^3 * Ck[n + 1]

    for k in range(n; stop=2, step=-1)

        Ck[k], Dk[k] = Bk[k] * [Ck[k + 1], Dk[k + 1]]
    end

    # find out in what layer `point` is
    pos = 0
    for k in 1:(n + 1)
        r < a[k] && (pos = k)
    end

    if pos == n + 1
        C = 0
        D = 0
    else
        C = Ck[pos + 1]
        D = Dk[pos + 1]
    end

    return C, D
end

"""
    scatteredfield(sphere::Hemispheres, excitation::UniformField, point, quantity::Field; parameter::Parameter=Parameter())

Compute the electric field scattered by a sphere where the upper and lower hemisphere have different permittivities, for an incident uniform field with polarization in the given direction
`Kettunen et al., 2007, Polarizability of a dielectric hemisphere`.

The point and returned field are in Cartesian coordinates.
"""
function scatteredfield(sphere::Hemispheres, excitation::UniformField, quantity::Field; parameter::Parameter=Parameter())

    Bz, Cz, Dz = scatterCoeffz(sphere, excitation, parameter.nmax)
    Bxy, Cxy, Dxy = scatterCoeffxy(sphere, excitation, parameter.nmax)

    F = zeros(fieldType(quantity), size(quantity.locations))

    # --- compute field in Cartesian representation
    for (ind, point) in enumerate(quantity.locations)
        F[ind] = scatteredfield(sphere, excitation, point, quantity, Bz, Cz, Dz, Bxy, Cxy, Dxy; parameter=parameter)
    end

    return F
end

function scatteredfield(
    sphere::Hemispheres{LR,LC},
    excitation::UniformField{FC,FT,FR},
    point,
    quantity::ScalarPotential,
    Bz,
    Cz,
    Dz,
    Bxy,
    Cxy,
    Dxy;
    parameter::Parameter=Parameter(),
) where {LR,LC,FC,FT,FR}

    E0 = excitation.amplitude
    dir = excitation.direction

    zproj = dot(dir, SVector(0.0, 0.0, 1.0))
    xproj = dot(dir, SVector(1.0, 0.0, 0.0))
    yproj = dot(dir, SVector(0.0, 1.0, 0.0))

    N = parameter.nmax

    r, θ, φ = cart2sph(point)

    if r < sphere.radii
        if θ < π / 2
            return zproj * [r^i * Pl(cos(θ), i) for i in 0:(N - 1)] ⋅ Cz + (xproj * cos(φ) + yproj * sin(φ)) * [r^i * Plm(cos(θ), i, 1) for i in 1:(N - 1)] ⋅ Cxy -
                   field(excitation, point, quantity; parameter=parameter)
        else
            return zproj * [r^i * Pl(cos(θ), i) for i in 0:(N - 1)] ⋅ Dz + (xproj * cos(φ) + yproj * sin(φ)) * [r^i * Plm(cos(θ), i, 1) for i in 1:(N - 1)] ⋅ Dxy -
                   field(excitation, point, quantity; parameter=parameter)
        end
    else
        return zproj * [r^(-(i + 1)) * Pl(cos(θ), i) for i in 0:(N - 1)] ⋅ Bz + (xproj * cos(φ) + yproj * sin(φ)) * [r^(-(i + 1)) * Plm(cos(θ), i, 1) for i in 1:(N - 1)] ⋅ Bxy
    end
end

function scatteredfield(
    sphere::Hemispheres{LR,LC},
    excitation::UniformField{FC,FT,FR},
    point,
    quantity::ElectricField,
    Bz,
    Cz,
    Dz,
    Bxy,
    Cxy,
    Dxy;
    parameter::Parameter=Parameter(),
) where {LR,LC,FC,FT,FR}

    E0 = excitation.amplitude
    dir = excitation.direction
    x̂ = SVector(1.0, 0.0, 0.0)
    ŷ = SVector(0.0, 1.0, 0.0)
    ẑ = SVector(0.0, 0.0, 1.0)
    zproj = dot(dir, ẑ)
    xproj = dot(dir, x̂)
    yproj = dot(dir, ŷ)

    N = parameter.nmax

    r, θ, φ = cart2sph(point)
    r̂ = sin(θ) * cos(φ) * x̂ + sin(θ) * sin(φ) * ŷ + cos(θ) * ẑ
    θ̂ = cos(θ) * cos(φ) * x̂ + cos(θ) * sin(φ) * ŷ + -sin(θ) * ẑ
    φ̂ = -sin(φ) * x̂ + cos(φ) * ŷ

    if r <= sphere.radii
        if θ < π / 2
            return -zproj * (
                [i * r^(i - 1) * Pl(cos(θ), i) for i in 1:(N - 1)] ⋅ Cz[2:end] * r̂ + (
                    if θ == 0.0
                        SVector(0.0, 0.0, 0.0)
                    else
                        [-r^(i - 1) * (i + 1) / sin(θ) * (cos(θ) * Pl(cos(θ), i) - Pl(cos(θ), i + 1)) for i in 0:(N - 1)] ⋅ Cz * θ̂
                    end
                )
            ) -
                   xproj * (
                [i * r^(i - 1) * Plm(cos(θ), i, 1) * cos(φ) for i in 1:(N - 1)] ⋅ Cxy * r̂ - (
                    if θ == 0.0
                        [
                            r^(i - 1) *
                            ((i + 1) * cos(θ) * (-1 / 2) * i * (1 + i) * cos(φ) - i * (-1 / 2) * (i + 1) * (2 + i) * cos(φ)) for
                            i in 1:(N - 1)
                        ] ⋅ Cxy * θ̂
                    else
                        [
                            r^(i - 1) / sin(θ) * ((i + 1) * cos(θ) * Plm(cos(θ), i, 1) * cos(φ) - i * Plm(cos(θ), i + 1, 1) * cos(φ))
                            for i in 1:(N - 1)
                        ] ⋅ Cxy * θ̂
                    end
                ) - (
                    if φ == 0.0 || φ == π
                        SVector(0.0, 0.0, 0.0)
                    else
                        [r^(i - 1) / sin(θ) * Plm(cos(θ), i, 1) * sin(φ) for i in 1:(N - 1)] ⋅ Cxy * φ̂
                    end
                )
            ) -
                   yproj * (
                [i * r^(i - 1) * Plm(cos(θ), i, 1) * sin(φ) for i in 1:(N - 1)] ⋅ Cxy * r̂ - (
                    if θ == 0.0
                        [
                            r^(i - 1) *
                            ((i + 1) * cos(θ) * (-1 / 2) * i * (1 + i) * sin(φ) - i * (-1 / 2) * (i + 1) * (2 + i) * sin(φ)) for
                            i in 1:(N - 1)
                        ] ⋅ Cxy * θ̂
                    else
                        [
                            r^(i - 1) / sin(θ) * ((i + 1) * cos(θ) * Plm(cos(θ), i, 1) * sin(φ) - i * Plm(cos(θ), i + 1, 1) * sin(φ))
                            for i in 1:(N - 1)
                        ] ⋅ Cxy * θ̂
                    end
                ) + (
                    if θ == 0.0
                        [r^(i - 1) * cos(φ) * (-1 / 2) * i * (i + 1) for i in 1:(N - 1)] ⋅ Cxy * φ̂
                    else
                        [r^(i - 1) / sin(θ) * Plm(cos(θ), i, 1) * cos(φ) for i in 1:(N - 1)] ⋅ Cxy * φ̂
                    end
                )
            ) - field(excitation, point, quantity; parameter=parameter)
        else
            return -zproj * (
                [i * r^(i - 1) * Pl(cos(θ), i) for i in 1:(N - 1)] ⋅ Dz[2:end] * r̂ + (
                    if θ ≈ π
                        SVector(0.0, 0.0, 0.0)
                    else
                        [-r^(i - 1) * (i + 1) / sin(θ) * (cos(θ) * Pl(cos(θ), i) - Pl(cos(θ), i + 1)) for i in 0:(N - 1)] ⋅ Dz * θ̂
                    end
                )
            ) -
                   xproj * (
                [i * r^(i - 1) * Plm(cos(θ), i, 1) * cos(φ) for i in 1:(N - 1)] ⋅ Dxy * r̂ - (
                    if θ ≈ π
                        [
                            r^(i - 1) * (
                                (i + 1) * cos(θ) * (-1 / 2) * i * (1 + i) * (-1)^(i + 1) * cos(φ) -
                                i * (-1 / 2) * (-1)^(i) * (i + 1) * (2 + i) * cos(φ)
                            ) for i in 1:(N - 1)
                        ] ⋅ Dxy * θ̂
                    else
                        [
                            r^(i - 1) / sin(θ) * (cos(θ) * (i + 1) * Plm(cos(θ), i, 1) * cos(φ) - i * Plm(cos(θ), i + 1, 1) * cos(φ))
                            for i in 1:(N - 1)
                        ] ⋅ Dxy * θ̂
                    end
                ) - (
                    if φ == 0.0 || φ ≈ π
                        SVector(0.0, 0.0, 0.0)
                    else
                        [r^(i - 1) / sin(θ) * Plm(cos(θ), i, 1) * sin(φ) for i in 1:(N - 1)] ⋅ Dxy * φ̂
                    end
                )
            ) -
                   yproj * (
                [i * r^(i - 1) * Plm(cos(θ), i, 1) * sin(φ) for i in 1:(N - 1)] ⋅ Dxy * r̂ - (
                    if θ ≈ π
                        [
                            r^(i - 1) * (
                                (i + 1) * cos(θ) * (-1 / 2) * i * (1 + i) * (-1)^(i + 1) * sin(φ) -
                                i * (-1 / 2) * (-1)^(i) * (i + 1) * (2 + i) * sin(φ)
                            ) for i in 1:(N - 1)
                        ] ⋅ Dxy * θ̂
                    else
                        [
                            r^(i - 1) / sin(θ) * (cos(θ) * (i + 1) * Plm(cos(θ), i, 1) * sin(φ) - i * Plm(cos(θ), i + 1, 1) * sin(φ))
                            for i in 1:(N - 1)
                        ] ⋅ Dxy * θ̂
                    end
                ) + (
                    if θ ≈ π
                        [r^(i - 1) * cos(φ) * (-1 / 2) * i * (-1)^(i + 1) * (i + 1) for i in 1:(N - 1)] ⋅ Dxy * φ̂
                    else
                        [r^(i - 1) / sin(θ) * Plm(cos(θ), i, 1) * cos(φ) for i in 1:(N - 1)] ⋅ Dxy * φ̂
                    end
                )
            ) - field(excitation, point, quantity; parameter=parameter)
        end
    else
        return -zproj * (
            [-(i + 1) * r^(-(i + 2)) * Pl(cos(θ), i) for i in 0:(N - 1)] ⋅ Bz * r̂ + (
                if θ == 0.0 || θ ≈ π
                    SVector(0.0, 0.0, 0.0)
                else
                    [-r^(-(i + 2)) * (i + 1) / sin(θ) * (cos(θ) * Pl(cos(θ), i) - Pl(cos(θ), i + 1)) for i in 0:(N - 1)] ⋅ Bz * θ̂
                end
            )
        ) -
               xproj * (
            [-(i + 1) * r^(-(i + 2)) * Plm(cos(θ), i, 1) * cos(φ) for i in 1:(N - 1)] ⋅ Bxy * r̂ +
            (
                if θ == 0.0 || θ ≈ π
                    [
                        -r^(-(i + 2)) * (
                            (i + 1) * cos(θ) * (-1 / 2) * i * cos(θ)^(i + 1) * (1 + i) * cos(φ) -
                            i * (-1 / 2) * cos(θ)^(i) * (i + 1) * (2 + i) * cos(φ)
                        ) for i in 1:(N - 1)
                    ] ⋅ Bxy * θ̂
                else
                    [
                        -r^(-(i + 2)) / sin(θ) * ((i + 1) * cos(θ) * Plm(cos(θ), i, 1) - i * Plm(cos(θ), i + 1, 1)) * cos(φ) for
                        i in 1:(N - 1)
                    ] ⋅ Bxy * θ̂
                end
            ) +
            (
                if φ == 0.0 || φ ≈ π
                    SVector(0.0, 0.0, 0.0)
                else
                    [-r^(-(i + 2)) / sin(θ) * Plm(cos(θ), i, 1) * sin(φ) for i in 1:(N - 1)] ⋅ Bxy * φ̂
                end
            )
        ) -
               yproj * (
            [-(i + 1) * r^(-(i + 2)) * Plm(cos(θ), i, 1) * sin(φ) for i in 1:(N - 1)] ⋅ Bxy * r̂ +
            (
                if θ == 0.0 || θ ≈ π
                    SVector(0.0, 0.0, 0.0)
                else
                    [
                        -r^(-(i + 2)) / sin(θ) * ((i + 1) * cos(θ) * Plm(cos(θ), i, 1) * sin(φ) - i * Plm(cos(θ), i + 1, 1) * sin(φ))
                        for i in 1:(N - 1)
                    ] ⋅ Bxy * θ̂
                end
            ) +
            (
                if θ == 0.0 || θ ≈ π
                    [r^(-(i + 2)) * cos(φ) * (-1 / 2) * i * cos(θ)^(i + 1) * (i + 1) for i in 1:(N - 1)] ⋅ Bxy * φ̂
                else
                    [r^(-(i + 2)) / sin(θ) * Plm(cos(θ), i, 1) * cos(φ) for i in 1:(N - 1)] ⋅ Bxy * φ̂
                end
            )
        )
    end
end

"""
    scatterCoeffz(sphere::Hemispheres, excitation::UniformField, N)

    Compute the coefficients for the different regions following `Kettunen et al., 2007, Polarizability of a dielectric hemisphere`
    for incident fields in z-direction.
    
"""
function scatterCoeffz(sphere::Hemispheres{LR,LC}, excitation::UniformField{FC,FT,FR}, N) where {LR,LC,FC,FT,FR}

    a = sphere.radii
    E0 = excitation.amplitude
    perms = getfield.(vcat(sphere.filling, excitation.embedding), 1)

    T = promote_type(LR, LC, FC, FT, FR)

    Σ = zeros(N)
    Σ[1] = SpecialFunctions.gamma(1) / SpecialFunctions.gamma(1 / 2)
    Σ[2] = SpecialFunctions.gamma(3 / 2) / SpecialFunctions.gamma(1)
    for (i, n) in enumerate(2:(N - 1))
        Σ[i + 2] = Σ[i] * (n) / (n - 1)
    end

    U = Matrix{T}(undef, N, N)

    for n in 0:(N - 1)
        for k in 0:(N - 1)
            if n == k
                U[n + 1, k + 1] = 1 / (2 * n + 1)
            elseif (n + k) % 2 == 0
                U[n + 1, k + 1] = 0.0
            else
                Ank = Σ[n + 1] / Σ[k + 1]
                Akn = 1 / Ank
                U[n + 1, k + 1] =
                    2 / π * (
                        sin(π / 2 * n) * cos(π / 2 * k) * Ank / (n * (n + 1) - k * (k + 1)) -
                        sin(π / 2 * k) * cos(π / 2 * n) * Akn / (n * (n + 1) - k * (k + 1))
                    )
            end
        end
    end

    eps1 = perms[1]
    eps2 = perms[2]
    ε0 = perms[3]

    M = Matrix{T}(undef, N, N)
    for k in 0:(N - 1)
        for n in 0:(N - 1)
            if k % 2 == 0
                η = 1
            else
                η = eps2 / eps1
            end
            M[k + 1, n + 1] =
                a^(-(n + 2)) *
                (η * (n + 1) + η * k * eps1 / ε0 + (-1)^(n + k) * (n + 1) + (-1)^(n + k) * k * eps2 / ε0) *
                U[n + 1, k + 1]
        end
    end
    A = Vector{T}(undef, N)
    for k in 0:(N - 1)
        if k % 2 == 0
            η = 1
        else
            η = eps2 / eps1
        end
        A[k + 1] = E0 * (η * k * eps1 / ε0 - η + (-1)^(1 + k) * k * eps2 / ε0 - (-1)^(1 + k)) * U[1 + 1, k + 1]
    end
    M2 = Matrix{T}(undef, N, N)
    for k in 0:(N - 1)
        for n in 0:(N - 1)
            if n % 2 == 0
                η = 1
            else
                η = eps2 / eps1
            end
            M2[k + 1, n + 1] =
                a^(n) * (η + η * (n * eps1 / (k + 1)) + (-1)^(n + k) + (-1)^(n + k) * (n * eps2) / (k + 1)) * U[n + 1, k + 1]
        end
    end
    A2 = Vector{T}(undef, N)
    for k in 0:(N - 1)
        A2[k + 1] = -E0 * a * (1 / (k + 1) + 1 + (-1)^(1 + k) / (k + 1) + (-1)^(1 + k)) * U[1 + 1, k + 1]
    end

    B = M \ A
    D = M2 \ A2
    C = Vector{T}(undef, N)
    for k in 0:(N - 1)
        if k % 2 == 0
            η = 1
        else
            η = eps2 / eps1
        end
        C[k + 1] = η * D[k + 1]
    end

    return B, C, D

end

"""
    scatterCoeffxy(sphere::Hemispheres, excitation::UniformField, N)

    Compute the coefficients for the different regions following `Kettunen et al., 2007, Polarizability of a dielectric hemisphere`
    for incident fields in x- or y-direction.
    
"""
function scatterCoeffxy(sphere::Hemispheres{LR,LC}, excitation::UniformField{FC,FT,FR}, N) where {LR,LC,FC,FT,FR}

    a = sphere.radii
    E0 = excitation.amplitude
    perms = getfield.(vcat(sphere.filling, excitation.embedding), 1)

    T = promote_type(LR, LC, FC, FT, FR)

    Σ = zeros(N)
    Σ[1] = SpecialFunctions.gamma(1) / SpecialFunctions.gamma(1 / 2)
    Σ[2] = SpecialFunctions.gamma(3 / 2) / SpecialFunctions.gamma(1)
    for (i, n) in enumerate(2:(N - 1))
        Σ[i + 2] = Σ[i] * (n) / (n - 1)
    end

    U = Matrix{T}(undef, N - 1, N - 1)

    for n in 1:(N - 1)
        for k in 1:(N - 1)
            if n == k
                U[n, k] = (n * (n + 1)) / (2 * n + 1)
            elseif (n + k) % 2 == 0
                U[n, k] = 0.0
            else
                Ank = Σ[n + 1] / Σ[k + 1]
                Akn = 1 / Ank
                U[n, k] =
                    2 / π * (
                        k * (k + 1) * sin(π / 2 * n) * cos(π / 2 * k) * Ank / (n * (n + 1) - k * (k + 1)) -
                        n * (n + 1) * sin(π / 2 * k) * cos(π / 2 * n) * Akn / (n * (n + 1) - k * (k + 1))
                    )
            end
        end
    end

    eps1 = perms[1]
    eps2 = perms[2]
    ε0 = perms[3]

    M = Matrix{T}(undef, N - 1, N - 1)
    for k in 1:(N - 1)
        for n in 1:(N - 1)
            if k % 2 == 0
                η = eps2 / eps1
            else
                η = 1
            end
            M[k, n] =
                a^(-(n + 2)) * (η * (n + 1) + η * k * eps1 / ε0 + (-1)^(n + k) * (n + 1) + (-1)^(n + k) * k * eps2 / ε0) * U[n, k]
        end
    end
    A = Vector{T}(undef, N - 1)
    for k in 1:(N - 1)
        if k % 2 == 0
            η = eps2 / eps1
        else
            η = 1
        end
        A[k] = -E0 * (η * k * eps1 / ε0 - η + (-1)^(1 + k) * k * eps2 / ε0 - (-1)^(1 + k)) * U[1, k]
    end
    M2 = Matrix{T}(undef, N - 1, N - 1)
    for k in 1:(N - 1)
        for n in 1:(N - 1)
            if n % 2 == 0
                η = eps2 / eps1
            else
                η = 1
            end
            M2[k, n] = a^(n) * (η + η * (n * eps1 / (k + 1)) + (-1)^(n + k) + (-1)^(n + k) * (n * eps2) / (k + 1)) * U[n, k]
        end
    end
    A2 = Vector{T}(undef, N - 1)
    for k in 1:(N - 1)
        A2[k] = E0 * a * (1 / (k + 1) + 1 + (-1)^(1 + k) / (k + 1) + (-1)^(1 + k)) * U[1, k]
    end

    B = M \ A
    D = M2 \ A2
    C = Vector{T}(undef, N - 1)
    for k in 1:(N - 1)
        if k % 2 == 0
            η = eps2 / eps1
        else
            η = 1
        end
        C[k] = η * D[k]
    end
    return B, C, D

end

fieldType(F::ElectricField)       = SVector{3,Complex{eltype(F.locations[1])}}
fieldType(F::DisplacementField)   = SVector{3,Complex{eltype(F.locations[1])}}
fieldType(F::ScalarPotential)     = Complex{eltype(F.locations[1])}
fieldType(F::ScalarPotentialJump) = Complex{eltype(F.locations[1])}
