# Pack a macro block of `fcount` free coordinates (offsets `fbuf`, enumerated
# by a split group, whole `E x L` groups) by `kc` K steps into the panels the
# per-sliver packer would write, reading each group line by line: coordinate
# `i0 + E * y2` is element `y2` of the line at `i0`. `@noinline`: one call per
# block, and inlined into the nest it slows the nest's micro-kernel loop.
@noinline function _pack_block_transposed!(
        plane_offset::PO, format::FMT, buffer::PK, descriptor::DS, ::Val{R}, Rp::Int,
        storage::ST, base::Int, fbuf::Vector{Int}, kaxis::KA, transform::F,
        fcount::Int, kc::Int, split
    ) where {PO, FMT <: PackFormat, PK, DS, R, ST, KA <: Axis, F}
    E, L = Int(split.E), Int(split.L)
    G = E * L
    # The sliver and lane come from a per-element divrem by the constant `R`;
    # hoisting them per line lets LLVM rewrite the loop so the line misses no
    # longer overlap.
    emit(kbase, i, p) = @inbounds _pack_line_element!(
        plane_offset, format, buffer, descriptor, Val(R), Rp * kc, transform(storage[kbase + fbuf[i + 1]]), i, p
    )
    if split.kinner
        for g0 in 0:G:(fcount - 1), y1 in 0:(E - 1), p in 0:(kc - 1)
            kbase = base + axis_offset(kaxis, p) + 1
            for y2 in 0:(L - 1)
                emit(kbase, g0 + y1 + E * y2, p)
            end
        end
    else
        for g0 in 0:G:(fcount - 1), p in 0:(kc - 1)
            kbase = base + axis_offset(kaxis, p) + 1
            for y1 in 0:(E - 1), y2 in 0:(L - 1)
                emit(kbase, g0 + y1 + E * y2, p)
            end
        end
    end
    valid = fcount % R
    if valid != 0
        rb = (fcount ÷ R) * Rp * kc
        pad = (plane, idx, pp) -> rb + plane_offset(descriptor, plane, idx, pp)
        for p in 0:(kc - 1), t in valid:(R - 1)
            _pack_emit_zero!(buffer, format, pad, t, p, realtype(descriptor))
        end
    end
    return nothing
end

# Element `x` of block coordinate `i` (sliver `i ÷ R`, lane `i % R`) at K step `p`.
@inline function _pack_line_element!(
        plane_offset::PO, format::FMT, buffer::PK, descriptor::DS, ::Val{R}, panel::Int, x, i::Int, p::Int
    ) where {PO, FMT, PK, DS, R}
    r, t = divrem(i, R)
    po = (plane, idx, pp) -> r * panel + plane_offset(descriptor, plane, idx, pp)
    _pack_emit!(buffer, format, po, t, p, convert(_element_type(format, scalartype(descriptor)), x))
    return nothing
end
