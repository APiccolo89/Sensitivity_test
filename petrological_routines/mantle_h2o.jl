using MAGEMin_C, HDF5, PyCall, Conda,CairoMakie
using Base.Threads
@pyimport scipy.interpolate as si

# ------------------------------------------------------------
function plot_phase_diagram(name::String; dir::String = @__DIR__,
                            field::String = "H2O_grid", label::String = "H2O in solids [wt%]")
    h5file = joinpath(dir, "$(name)_phase_diagram.h5")
    T, P, Z = h5open(h5file, "r") do f
        read(f, "T"), read(f, "P"), read(f, field)       # P already in GPa (written as P./10)
    end
    size(Z) == (length(T), length(P)) || (Z = permutedims(Z))   # guard against a transposed grid

    outdir = joinpath(dir, "phase_diagrams")
    mkpath(outdir)

    fig = Figure(size = (720, 600))
    ax  = Axis(fig[1, 1], xlabel = "T [°C]", ylabel = "P [GPa]",
               title = "$(name) – $(label)", yreversed = true)
    hm  = heatmap!(ax, T, P, Z, colormap = :solar, colorrange = (minimum(Z), maximum(Z)))
    Colorbar(fig[1, 2], hm, label = label)

    out = joinpath(outdir, "$(name)_$(field).png")
    save(out, fig, px_per_unit = 2)
    return out
end

#------------------------------------------------------------
function mantle_h2O(data::MAGEMin_Data, X::Array{Float64,1}, P::Float64, T::Float64)
    Xoxides = ["SiO2", "Al2O3", "MgO", "FeO", "O","H2O", "S", "CaO", "Na2O"]
    out     = single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in="wt",name_solvus=true)

    ant = 0.0
    bru = 0.0
    chl = 0.0
    for (i) in 1:length(out.ph)
        name = out.ph[i] 
        if name == "atg"
            ant = out.ph_frac_vol[i]
        elseif name == "br"
            bru = out.ph_frac_vol[i]
        elseif name == "chl"
            chl = out.ph_frac_vol[i]
        end
    end

    H = out.enthalpy[1]


    return out.frac_F_wt, out.bulk_S_wt, out.rho, ant, bru, chl, H
end
#------------------------------------------------------------
function extract_data(file_path::String)
    file = h5open(file_path, "r")

    coord = read(file["/Mesh/mesh/geometry"])

    x = coord[1,:];
    y = coord[2,:];
    P = read(file["/Function/Lit Pres  [GPa]/0"]);
    T = read(file["/Function/Temperature  [degC]/0"]);
    v = read(file["/Function/Velocity  [cm/yr]/0"]);
    vx = v[1,:]*1e-5/3.15e7; # convert from cm/yr to km/s
    vy = v[2,:]*1e-5/3.15e7; # convert from cm/yr to km/s

    return x, y, P, T, vx, vy
end
#------------------------------------------------------------
function create_phase_diagram(data::MAGEMin_Data, X::Array{Float64,1},name::String,water_adjustment::Bool)
    T = 300.0:5.0:1200.0
    P = 5.0:0.5:80
    
    fl_grid = zeros(length(T), length(P))
    rho_grid = zeros(length(T), length(P))
    H2O_grid = zeros(length(T), length(P))
    chl = zeros(length(T), length(P))
    bru = zeros(length(T), length(P))
    ant = zeros(length(T), length(P))
    enthalpy = zeros(length(T), length(P))

    t0 = time()
    for i in 1:length(T)
        
        for j in 1:length(P)
            fl_grid[i,j], X_solid ,rho_grid[i,j], ant[i,j], bru[i,j], chl[i,j],enthalpy[i,j] = mantle_h2O(data, X, P[j], T[i])
            H2O_grid[i,j] = X_solid[6]
        end     
        
    end
    println("Phase diagram for $(name) completed in $(time()-t0) seconds.")
    println("============================================================")

    # Save the phase diagram data to an HDF5 file
    h5open(joinpath(@__DIR__, "$(name)_phase_diagram.h5"), "w") do file
        write(file, "T", collect(T))
        write(file, "P", collect(P)./10)
        write(file, "fl_grid", fl_grid)
        write(file, "rho_grid", rho_grid)
        write(file, "H2O_grid", H2O_grid)
        write(file, "chl", chl)
        write(file, "bru", bru)
        write(file, "ant", ant)
        write(file, "enthalpy", enthalpy)
    end

    for (fld, lab) in (("H2O_grid", "H2O in solids [wt%]"),
                   ("fl_grid",  "fluid fraction [wt frac]"),
                   ("rho_grid", "density [kg m⁻³]"),
                   ("chl", "Chlorite fraction [wt frac]"),
                   ("bru", "Brucite fraction [wt frac]"),
                   ("ant", "Antigorite fraction [wt frac]"),
                   ("enthalpy", "Enthalpy [J/kg]"))
        println("Plotting phase diagram for $(fld) of $(name)...")
        plot_phase_diagram(name, field = fld, label = lab)
    end

end
#------------------------------------------------------------
mutable struct Points
           x::Array{Float64,1}
           y::Array{Float64,1}
           vx::Array{Float64,1}
           vy::Array{Float64,1}
           P::Array{Float64,1}
           T::Array{Float64,1}
           fl::Array{Float64,1}
           fl_ex::Array{Float64,1}
           df::Array{Float64,1}
           X::Array{Float64,2}
           H2O::Array{Float64,1}
end
#------------------------------------------------------------
function add_water(X_dry::Vector{Float64}, w::Float64)   # X_dry without H2O, wt
    """
    Add water to a dry composition: use the dry composition, compute the weight fraction of each
    of the element, such that the total weight fraction of the dry composition is 1-w, and the weight fraction of water is w.
    The function returns the new composition with water added, and the weight fraction of water. 
    """
    
    X = X_dry ./ sum(X_dry) .* (100.0 * (1 - w))   
    X[6] = w*100.0        # dry part sums to 100(1-w)
    return X     # oxides, H2O
end
#------------------------------------------------------------
function main()
    data = Initialize_MAGEMin("um",verbose=false);

    X_mantle = [47.1100; 2.6800; 36.49000; 7.5200; 0.0000;  0.0000; 0.00000; 4.2000; 0.00000];
    X_Basalt = [49.0800; 15.2300;7.37000;10.2800;0.0000;  3.10000; 0.00000; 10.7700; 2.52000];
    X_Gabbro =  [51.01;16.23;9.27;6.25; 0.00; 0.80; 0.00; 12.60; 2.82]
    # Normalize the compositions to 100 wt%
    X_mantle .= X_mantle./sum(X_mantle).*100.0
    X_Basalt .= X_Basalt./sum(X_Basalt).*100.0
    X_Gabbro .= X_Gabbro./sum(X_Gabbro).*100.0

    # Change X_Mantle composition to have 2 wt% to 4 wt% H2O
    x_wt_2 = add_water(X_mantle, 0.02)
    x_wt_4 = add_water(X_mantle, 0.04)

    # Create Phase diagram of the initial composition 
    create_phase_diagram(data, X_mantle, "Mantle", false)
    create_phase_diagram(data, x_wt_2, "Mantle_wt_2", false)
    create_phase_diagram(data, x_wt_4, "Mantle_wt_4", false)
    create_phase_diagram(data, X_Basalt, "Basalt", false)
    create_phase_diagram(data, X_Gabbro, "Gabbro", false)


end
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
# Main Function
main()