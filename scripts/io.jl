using CSV

"""
    read_final_events(filepath::AbstractString)

Function for reading the final events from the GiBUU output  

# Arguments
- `filepath::AbstractString`: filepath to the FinalEvents.dat
"""
function read_final_events(filepath::AbstractString)
    file = open(filepath)
    header = readline(file)
    close(file)
    raw_col_names = split(header)[2:end]
    col_names = [String.(split(col,":"))[end] for col in raw_col_names]
    CSV.read(filepath, 
             header=col_names, 
             delim=' ', 
             comment="#", 
             ignorerepeated=true, 
             types=[Int32, 
                    Int32, 
                    Int32, 
                    Int32, 
                    Float64,
                    Float64,
                    Float64,
                    Float64,
                    Float64,
                    Float64,
                    Float64,
                    Float64,
                    Int32, 
                    Int32, 
                    Float64
                    ])
end


