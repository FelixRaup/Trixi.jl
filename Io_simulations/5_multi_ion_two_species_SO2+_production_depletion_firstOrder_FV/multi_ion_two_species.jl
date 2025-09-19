
using OrdinaryDiffEq #LowStorageRK
using Trixi

# define new structs inside a module to allow re-evaluating the file
# This is done to create a refinement indicator that keeps the solution refined at the
# immersed boundary
module TrixiExtension

using Trixi
#using StaticArrays

struct VerticalIndicator{RealT <: Real, IndicatorCache}
    center::Array{RealT, 1}
    radius::RealT
    cache::IndicatorCache
end

function VerticalIndicator(center, radius)
    cache = create_horizontal_cache(typeof(radius))
    return VerticalIndicator(center, radius, cache)
end

function create_horizontal_cache(RealT)
    alpha = Vector{RealT}()
    cache = (; alpha)
    return cache
end

function get_element_variables!(element_variables, indicator::VerticalIndicator, amr_callback)
    element_variables[:indicator_amr] = indicator.cache.alpha
end

# Refine the elements inside the moon and the transition zones between the plasma torus, the low-density zone, and the Jovian ionosphere
function (vertical_indicator::VerticalIndicator)(
    u::AbstractArray, mesh, equations, dg::DGSEM, cache; kwargs...)

    @unpack alpha = vertical_indicator.cache
    @unpack radius, center = vertical_indicator
    @unpack node_coordinates = cache.elements
    resize!(alpha, nelements(dg, cache))

    Trixi.@threaded for element in eachelement(dg, cache)
        alpha[element] = 0.0
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            x = Trixi.get_node_coords(node_coordinates, equations, dg, i, j, k, element)

            # Moon surroundings and Galileo trajectory
            distance = sqrt(x[2]^2 + x[3]^2)
            if distance < radius && -7.5 < x[1] < 7
                alpha[element] = 1.0
            end

            # # Transition between plasma torus and low-density region
            # r = sqrt(x[3] ^ 2 + x[2] ^ 2)
            # if (12 < r < 16) && abs(x[2]) < 5
            #     alpha[element] = 1.0
            # end

            # # Transition between low-density region and Jovian ionosphere
            # if (33 + eps(1.0) < abs(x[3]) < 45 - eps(1.0)) && abs(x[2]) < 5
            #     alpha[element] = 1.0
            # end
            center2 = [10.0, 0.0, 42.5]
            radius2 = 3.0
            distance = Trixi.norm(x - center2)
            if distance < radius2
                alpha[element] = 1.0
            end
            center3 = [10.0, 0.0, -42.5]
            radius3 = 3.0
            distance = Trixi.norm(x - center3)
            if distance < radius3
                alpha[element] = 1.0
            end

        end
    end

    return alpha
end

end # module TrixiExtension

import .TrixiExtension


###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

# equations = IdealGlmMhdMultiIonEquations3D(gammas = (5/3, 1.29),
#                                            charge_to_mass = (1.0, 0.0))

equations = IdealGlmMhdMultiIonEquations3D(gammas = (5/3, 1.29),
                                           charge_to_mass = (112.881325104939, 25.7781872234837),
                                           gas_constants = (1.0, 0.296875), # [nondimensional]
                                           molar_masses = (19, 64), # [nondimensional]
                                           ion_ion_collision_constants = [0.0 1.51108336025336E-07;
                                                                          1.51108336025336E-07 0.0], 
                                                                          )

# equations = IdealGlmMhdMultiIonEquations3D(gammas = (5/3, 5/3),
#                                         charge_to_mass = (112.9, 112.9))





function initial_condition_io_nondimensional(x, t, equations::IdealGlmMhdMultiIonEquations3D)

    B1 = 0.0
    B2 = 0.0
    B3 = -3.604

    rho_torus = 1.0
    v1_torus = 1.0
    v2_torus = 0.0
    v3_torus = 0.0
    p_torus = 0.148984037940128

    rho_SO2 = rho_torus * 1e-1 # 1.e-3
    v1_SO2 = 1.0
    v2_SO2 = 0.0
    v3_SO2 = 0.0
    p_SO2 = p_torus * 0.3 * rho_SO2/rho_torus # assuming pressure balance and same temp for torus and SO2+ 

    #p_SO2 = 1e-1 # 4.4E-03  
    # p_SO2 = 4.42296362634754E-05 # 100% of the torus' temperature for density SO2+ of 1e-3

    psi = 0.0

    # rho_io_surface = 1 # 1.51e7 # [nondimensional] ~ 1e10 cm^(-3)
    # p_io_surface = 0.1 # 4381.88346882728  # [nondimensional] ~ 1e-2 micro bars

    # rho_SO2_io_surface = rho_torus * 0.1
    # p_SO2_io_surface = p_torus * 0.3 * rho_SO2_io_surface/rho_torus

    # r = sqrt(sum(x .* x))
    # d = 150/1820 # dilatation factor that models how Io’s atmosphere becomes thinner away from the moon’s surface


    prim = SVector(B1, B2, B3, rho_torus, v1_torus, v2_torus, v3_torus, p_torus, rho_SO2, v1_SO2, v2_SO2, v3_SO2, p_SO2, psi)

    return prim2cons(prim, equations)
end



function source_terms_io_nondimensional(u, x, t, equations::IdealGlmMhdMultiIonEquations3D)

    # Ion-Neutral collisions (species 1)

    omega_in = 127.719 
    rho_io_surface = 1.81560284E+06

    r = sqrt(sum(x .* x))
    d = 150/1820

    if r <= one(real(equations))
        omega = omega_in 
        rho_io = 0 # rho_io_surface  
    else
        omega = omega_in * exp((one(real(equations)) - r) / d)
        rho_io = rho_io_surface * exp((one(real(equations)) - r) / d)
    end

    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(u, equations)

    # s_0 = zero(real(equations))
    s_0 = 0.0

    s_coll_torus_io_neutrals = SVector(s_0, s_0, s_0, s_0, -omega * u[5], -omega * u[6], -omega * u[7], -omega * (u[8] - 0.5f0 * (B1^2 + B2^2 + B3^2 + psi^2)), s_0, s_0, s_0, s_0, s_0, s_0)

    s_coll_SO2_plus_io_neutrals = SVector(s_0, s_0, s_0, s_0, s_0, s_0, s_0, s_0, s_0, -omega * u[10], -omega * u[11], -omega * u[12], -omega * (u[13] - 0.5f0 * (B1^2 + B2^2 + B3^2 + psi^2)), s_0)


    # Production of SO2+ (electron impact ionization with constant rate: f_SO2 does not depend on electron temperature)

    total_electron_charge = zero(eltype(u))
    for k in eachcomponent(equations)
        rho_k = u[3 + (k - 1) * 5 + 1]
        charge_ratio = rho_k * equations.charge_to_mass[k]
        total_electron_charge += charge_ratio
    end

    prim = cons2prim(u, equations)
    rho_SO2, v1_SO2, v2_SO2, v3_SO2, p_SO2 = Trixi.get_component(2, prim, equations)

    P_inv_ene_nn = 6.82218207E-08 # Scaled impact ionization rate = P / (e * n_e * n_n)

    s_SO2_rho = s_SO2_u = s_SO2_v = s_SO2_w = s_SO2_e = 0.0

    s_SO2_rho = P_inv_ene_nn * total_electron_charge * rho_io
    s_SO2_u += s_SO2_rho * v1_SO2   # Felix removed negative sign here
    s_SO2_v += s_SO2_rho * v2_SO2 
    s_SO2_w += s_SO2_rho * v3_SO2
    s_SO2_e += 0.5 * s_SO2_rho * (v1_SO2^2 + v2_SO2^2 + v3_SO2^2) # 0.5 * s_SO2_rho v^2 + v1*s_u+ v2*s_v+ v3*s_w

    s_SO2_creation = SVector(s_0, s_0, s_0, s_0, s_0, s_0, s_0, s_0, s_SO2_rho, s_SO2_u, s_SO2_v, s_SO2_w, s_SO2_e, s_0)


    # Depletion of S02+

    total_electron_charge = zero(eltype(u))
    for k in eachcomponent(equations)
        rho_k = u[3 + (k - 1) * 5 + 1]
        charge_ratio = rho_k * equations.charge_to_mass[k]
        total_electron_charge += charge_ratio
    end

    prim = cons2prim(u, equations)
    rho_SO2, v1_SO2, v2_SO2, v3_SO2, p_SO2 = Trixi.get_component(2, prim, equations)

    k_dep = 1e-6 # depletion rate ?

    # density change
    s_torus_rho = + k_dep * total_electron_charge * rho_SO2
    s_SO2_rho = - k_dep * total_electron_charge * rho_SO2

    # momentum change
    s_torus_p1 = s_SO2_rho * v1_SO2
    s_torus_p2 = s_SO2_rho * v2_SO2
    s_torus_p3 = s_SO2_rho * v3_SO2

    s_SO2_p1 = - s_SO2_rho * v1_SO2
    s_SO2_p2 = - s_SO2_rho * v2_SO2
    s_SO2_p3 = - s_SO2_rho * v3_SO2

    # total kinetic energy change 
    s_torus_e = 0.5 / s_SO2_rho * (s_SO2_p1^2 + s_SO2_p2^2 + s_SO2_p3^2)
    s_SO2_e = - 0.5 / s_SO2_rho * (s_SO2_p1^2 + s_SO2_p2^2 + s_SO2_p3^2)

    s_SO2_depletion = SVector(s_0, s_0, s_0, s_torus_rho, s_torus_p1, s_torus_p2, s_torus_p3, s_torus_e, s_SO2_rho, s_SO2_p1, s_SO2_p2, s_SO2_p3, s_SO2_e, s_0)


    # Ion-Ion collision
    s_ion_coll = source_terms_collision_ion_ion(u, x, t, equations)

    # Lorentz force
    s_lorentz = source_terms_lorentz(u, x, t, equations)


    return s_coll_torus_io_neutrals + s_coll_SO2_plus_io_neutrals + s_SO2_creation + s_SO2_depletion + s_lorentz + s_ion_coll
end




initial_condition = initial_condition_io_nondimensional

# surface_flux=(FluxPlusDissipation(flux_ruedaramirez_etal,
#                                                           DissipationLaxFriedrichsEntropyVariables()), flux_nonconservative_ruedaramirez_etal)

surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)

volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
                                         
# volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                  volume_flux_dg = volume_flux,
#                                                  volume_flux_fv = surface_flux)

volume_integral = VolumeIntegralPureLGLFiniteVolume(surface_flux) # 1. Order FV



solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)


# coordinates_min = (-20.0, -20.0, -50.0)
# coordinates_max = (60.0, 20.0, 50.0)

# trees_per_dimension = (8, 4, 10)
# mesh = P4estMesh(trees_per_dimension,
#                  polydeg = 1,
#                  coordinates_min = coordinates_min,
#                  coordinates_max = coordinates_max,
#                  initial_refinement_level = 0,
#                  periodicity = true)


coordinates_min = (-10.0, -10.0, -10.0)
coordinates_max = (10.0, 10.0, 10.0)

mesh = TreeMesh( 
                coordinates_min,
                coordinates_max,
                n_cells_max = 100000,
                periodicity = true,
                initial_refinement_level = 3,
                )

    
# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, 
                                    source_terms = source_terms_io_nondimensional,
                                    )

@inline function density1(u, equations::IdealGlmMhdMultiIonEquations3D)
    u_loc = Trixi.get_component(1, u, equations)
    return u_loc[1]
end

@inline function density2(u, equations::IdealGlmMhdMultiIonEquations3D)
    u_loc = Trixi.get_component(2, u, equations)
    return u_loc[1]
end

@inline function pressure1(u, equations::IdealGlmMhdMultiIonEquations3D)
    pres = pressure(u, equations)
    return pres[1]
end

@inline function pressure2(u, equations::IdealGlmMhdMultiIonEquations3D)
    pres = pressure(u, equations)
    return pres[2]
end

# positivity limiter necessary for this tough example
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6, 5.0e-6, 5.0e-6), # maybe lower p2 threshold
                                                     variables = (density1, density2, pressure1, pressure2))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0) #0.06) #
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
			             output_directory=joinpath(@__DIR__,"out"))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorLöhner(semi,
                                variable = density_pressure)

# An indicator that refines where the moon (surface) is
amr_indicator_moon = TrixiExtension.VerticalIndicator([0.0, 0.0, 0.0], 1.5)  #IndicatorImmersedBoundary(semi)

amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, amr_indicator_moon,
                                                base_level = 3,
                                                med_level = 5, med_threshold=0.2, #3
                                                max_level = 7, max_threshold = 0.25, #4 0.25
                                                max_threshold_secondary = 0.1)



amr_callback = AMRCallback(semi, amr_controller,
                            interval = 5,
                            adapt_initial_condition = true,
                            adapt_initial_condition_only_refine = true,
                            limiter! = stage_limiter!
                                                   )

cfl = 0.3
stepsize_callback = StepsizeCallback(cfl = cfl)


save_solution = SaveSolutionCallback(dt = 0.2, #interval = 1, #
                                     save_initial_solution=true,
                                     save_final_solution=true,
			                         output_directory=joinpath(@__DIR__,"out"),
                                     solution_variables=cons2prim)

# The SaveRestartCallback allows to save a file from which a Trixi.jl simulation can be restarted
save_restart = SaveRestartCallback(interval = 1000,
                                   output_directory=joinpath(@__DIR__,"out"),
                                   save_final_restart = true)

glm_speed_callback = GlmSpeedCallback(glm_scale = 1.0, cfl = cfl) # 

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        amr_callback,
                        stepsize_callback,
                        save_solution,
                        glm_speed_callback,
                        #save_restart
                        )



###############################################################################
# run the simulation

sol = solve(ode, SSPRK33(stage_limiter!); 
            dt = 1.0e-7,
            ode_default_options()..., 
            adaptive = false,
            callback = callbacks);


summary_callback() 