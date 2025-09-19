# open(joinpath(@__DIR__,"log_ec_n3_new.txt"), "w") do io
#     redirect_stdout(io) do
#         convergence_test(joinpath(@__DIR__,"elixir_convergence_ec.jl"), 3, polydeg=3, cells_per_dimension = (4 ,4, 4))
#     end
# end

# open(joinpath(@__DIR__,"log_llf_n3_new.txt"), "w") do io
#     redirect_stdout(io) do
#         convergence_test(joinpath(@__DIR__,"elixir_convergence_llf.jl"), 3, polydeg=3, initial_refinement_level = 1)
#     end
# end

open(joinpath(@__DIR__,"log_es_hmat_n3_new.txt"), "w") do io
    redirect_stdout(io) do
        convergence_test(joinpath(@__DIR__,"elixir_convergence_es_Hmat.jl"), 3, polydeg=3, cells_per_dimension = (4 ,4, 4))
    end
end
