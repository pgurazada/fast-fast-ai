function ele_mul(scalar, vector)

    out = [0, 0, 0]

    for i in 1:length(out)
        out[i] = scalar * vector[i]
    end

    return out
    
end
