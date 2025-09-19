
using OrdinaryDiffEq# LowStorageRK
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

# equations = IdealGlmMhdMultiIonEquations3D(gammas = (5/3),
#                                            charge_to_mass = (112.881325104939)
#                                             )

equations = IdealGlmMhdMultiIonEquations3D(gammas = (1.29),
                                           charge_to_mass = (25.7781872234837)
                                           )

"""
    initial_condition_io_nondimensional(x, t, equations::IdealGlmMhdMultiIonEquations3D)

Non-dimensional simplified setup for the simulation of the interaction of Io with Jupiter's
magnetosphere and plasma torus.
- A. M. Rueda-Ramírez, S. Hennemann, F. J. Hindenlang, A. R. Winters, G. J. Gassner (2021)
  An entropy stable nodal discontinuous Galerkin method for the resistive MHD
  equations. Part II: Subcell finite volume shock capturing
  [doi: 10.1016/j.jcp.2021.110580](https://doi.org/10.1016/j.jcp.2021.110580)
"""
function initial_condition_io_nondimensional(x, t, equations::IdealGlmMhdMultiIonEquations3D)

    # r = sqrt(x[3] ^ 2 + x[2] ^ 2)
    # z_abs = abs(x[3])
    # # Stratified densities
    # if r < 12 - eps(1.0)
    #     # Plasma torus
    #     rho = 1.0
    # elseif r < 16 - eps(1.0)
    #     # Transition between plasma Torus and low-density zone
    #     rho = -0.225 * r + 3.7
    # elseif z_abs < 33 - eps(1.0)
    #     # Low-density zone
    #     rho = 0.1
    # elseif z_abs < 45 - eps(1.0)
    #     # Transition between low-density zone and Jovian ionosphere
    #     rho = -2.87905093e-03 * z_abs^4 + 4.26099537e-01 * z_abs^3 - 2.33721354e+01 * z_abs^2+ 5.64351562e+02 * z_abs -5.06965820e+03
    # else # x[3] > 45
    #     rho = 20.0
    # end

    rho_torus = 1.0
    p_torus = 0.149

    # rho = rho_torus
    # p =  p_torus

    rho = 0.1 * rho_torus
    p = rho/rho_torus * 0.3 * p_torus

    prim = SVector(0.0, 0.0, -3.604, rho, 1.0, 0.0, 0.0, p, 0.0)

    return prim2cons(prim, equations)
end


function source_terms_io_nondimensional(u, x, t, equations::IdealGlmMhdMultiIonEquations3D)
    omega_in = 127.719
    r = sqrt(sum(x .* x))
    d = 150/1820
    if r <= one(real(equations))
        omega = omega_in
    else
        omega = omega_in * exp((one(real(equations)) - r) / d)
    end
    B1, B2, B3 = magnetic_field(u, equations)
    psi = divergence_cleaning_field(u, equations)

    return SVector(zero(real(equations)), zero(real(equations)), zero(real(equations)), zero(real(equations)), -omega * u[5], -omega * u[6], -omega * u[7], -omega * (u[8] - 0.5f0 * (B1^2 + B2^2 + B3^2 + psi^2)), zero(real(equations)))
end

initial_condition = initial_condition_io_nondimensional

surface_flux=(FluxPlusDissipation(flux_ruedaramirez_etal,
                                                          DissipationLaxFriedrichsEntropyVariables()), flux_nonconservative_ruedaramirez_etal)
volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)


coordinates_min = (-20.0, -20.0, -50.0)
coordinates_max = (60.0, 20.0, 50.0)

trees_per_dimension = (8, 4, 10)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 1,
                 coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 initial_refinement_level = 0,
                 periodicity = true)

# This is not working yet on Trixi
boundary_condition = BoundaryConditionDirichlet(initial_condition_io_nondimensional)
boundary_conditions = Dict(:x_neg => boundary_condition,
                           :x_pos => boundary_condition,
                           :y_neg => boundary_condition,
                           :y_pos => boundary_condition,
                           :z_neg => boundary_condition,
                           :z_pos => boundary_condition)
    
# create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, 
                                    source_terms = source_terms_io_nondimensional,
                                    #boundary_conditions = boundary_conditions
                                    )

# positivity limiter necessary for this tough example
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

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
# An indicator that refines where the moon is
amr_indicator_moon = TrixiExtension.VerticalIndicator([0.0, 0.0, 0.0], 1.5)  #IndicatorImmersedBoundary(semi)

amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, amr_indicator_moon,
                                                base_level = 0,
                                                med_level = 3, med_threshold=0.2, #3
                                                max_level = 4, max_threshold = 0.25, #4
                                                max_threshold_secondary = 0.1)
amr_callback = AMRCallback(semi, amr_controller,
                            interval = 5,
                            adapt_initial_condition = true,
                            adapt_initial_condition_only_refine = true,
                            #limiter! = stage_limiter!
                                                   )

cfl = 0.3
stepsize_callback = StepsizeCallback(cfl = cfl)


save_solution = SaveSolutionCallback(dt = 0.5, #interval = 1, #
                                     save_initial_solution=true,
                                     save_final_solution=true,
			                         output_directory=joinpath(@__DIR__,"out"),
                                     solution_variables=cons2prim)

# The SaveRestartCallback allows to save a file from which a Trixi.jl simulation can be restarted
save_restart = SaveRestartCallback(interval = 500,
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

sol = solve(ode, SSPRK33(stage_limiter!);
            dt = 1.0e-7,
            ode_default_options()..., 
            adaptive = false,
            callback = callbacks);

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
#             dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#             save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
