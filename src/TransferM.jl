#
#	apply TM 
#
function applyTM_OP(MPS_ket::AbstractArray{<:Number}, MPS_bra::AbstractArray{<:Number}, Op::localOp{DT,D}, 
		    x::AbstractArray{<:Number}; left::Bool = true) where {DT<:Number, D}
	
	res = zeros(eltype(x), size(MPS_bra)[[1,3][left+1]], size(MPS_ket)[[1,3][left+1]])
	tmp = zeros(eltype(x), size(x)[2], size(MPS_bra)[[1,3][left+1]])
	
	if left == true 
		@inbounds for op in eachoperation(Op)

			c = ComplexF64(op[1])
			mket = @view MPS_ket[:, op[2], :] #α_ket, β_ket
			mbra = @view MPS_bra[:, op[3], :] #α_bra, β_bra

			LinearAlgebra.BLAS.gemm!('T','N', c, x, conj.(mbra), ComplexF64(0.0), tmp)  #α_ket, β_bra
			LinearAlgebra.BLAS.gemm!('T', 'N', ComplexF64(1.0), tmp, mket, ComplexF64(1.0), res)
		end

	else
		
		@inbounds for op in eachoperation(Op)
			
			c = ComplexF64(op[1])
			mket = @view MPS_ket[:, op[2], :] #α_ket, β_ket
			mbra = @view MPS_bra[:, op[3], :] #α_bra, β_bra

			LinearAlgebra.BLAS.gemm!('T','T', c, x, conj.(mbra), ComplexF64(0.0), tmp)  #β_ket, α_bra
			LinearAlgebra.BLAS.gemm!('T', 'T', ComplexF64(1.0), tmp, mket, ComplexF64(1.0), res)
		end



	end
	
	return res
end

function applyTM_OP(MPSket::Vector{<:Any}, MPSbra::Vector{<:Any}, Op_string::NTuple{N,Int}, vecMPO::Vector{<:MPOsparseT}, 
 		  x::AbstractArray{<:Number}; left::Bool = true) where {N}
	
	# TODO check that MPSket is the same length as MPSbra and that phys dim match
	mps_chainLength = length(MPSket)
	mps_chainLength == length(MPSbra) || throw(ArgumentError("length of ket and bra MPS do not match."))
	mps_chainLength == N || throw(ArgumentError("MPO and MPS length do not match."))



	if left == true 
		A = MPSket[1]
        	B = MPSbra[1]
		local_op = vecMPO[1][Op_string[1]]
	else 
		A = MPSket[end]
		B = MPSbra[end]
		local_op = vecMPO[end][Op_string[end]]
	end


	res = left == true ? applyTM_OP(A, B, local_op, x; left = true) : applyTM_OP(A, B, local_op, x; left = false)

	# go thorugh unit cell
	if mps_chainLength > 1
		if left == true
			res = applyTM_OP(MPSket[2:end], MPSbra[2:end], Op_string[2:end], vecMPO[2:end], res; left = true)
		else
			res = applyTM_OP(MPSket[1:end-1], MPSbra[1:end-1], Op_string[1:end-1], vecMPO[1:end-1], res; left = false)
		end
	end

		
	return res # (α_bra , α_ket) 
end

function applyTM_MPO(MPSket::Vector{<:Any}, MPSbra::Vector{<:Any}, MPOvec::Vector{<:MPOsparseT}, x::Vector{<:Any}; left::Bool = true) where {DT<:Number, D}
	mps_chainLength = length(MPSket)
	mps_chainLength == length(MPSbra) || throw(ArgumentError("length of ket and bra MPS do not match."))
	mps_chainLength == length(MPOvec) || throw(ArgumentError("MPO and MPS length do not match."))

	#check if same length then MPOvec
	
	A = left == true ? MPSket[1] : MPSket[end]
	B = left == true ? MPSbra[1] : MPSbra[end]
	local_MPO = left == true ? MPOvec[1] : MPOvec[end]
	b_dim = length(local_MPO.bDim) == 1 ? (local_MPO.bDim, local_MPO.bDim) : local_MPO.bDim
	Op_idx = local_MPO.Op_index
	d_phys = size(MPSket[1])[2]
	
	if left == true
		YLa = Vector{Any}(undef,b_dim[2])
	
		for diag_idx = b_dim[2]:-1:1 #go through outgoing bond dim

                	a = diag_idx
			YLa[a] = zeros(ComplexF64,size(B)[3],size(A)[3])
		
			for b = 1:b_dim[1]
				
				Op = local_MPO[b, a]

				if Op != nothing

					tmp = x[b]		
					tmp = applyTM_OP(A, B, Op, tmp; left = true)
					YLa[a] = YLa[a] + tmp #(α_bra ,α_ket)	
				else

					YLa[a] = YLa[a] #(α_bra ,α_ket)
				end
                		
                                                                                                                              
                	end
                end
	else
		YRa = Vector{Any}(undef,b_dim[1])
		for diag_idx = 1:b_dim[1] 
			a = diag_idx
			YRa[a] = zeros(ComplexF64,size(B)[1],size(A)[1])
			
			for b = b_dim[2]:-1:1
				
				
				Op = local_MPO[a, b]
				
				if Op != nothing	
					
					tmp = x[b]		
					tmp = applyTM_OP(A, B, Op, tmp; left = false)
					YRa[a] = YRa[a] + tmp #(α_bra ,α_ket)	
				else

					YRa[a] = YRa[a] #(α_bra ,α_ket)
				end

			end		                                                                                 
		end                                                                                                      	
	
	end



	if mps_chainLength > 1
		if left == true
			YLa = applyTM_MPO(MPSket[2:end], MPSbra[2:end], MPOvec[2:end], YLa; left = true)
		else
			YRa = applyTM_MPO(MPSket[1:end-1], MPSbra[1:end-1], MPOvec[1:end-1], YRa; left = false)
		end
	
	end

	return res = left == true ? YLa : YRa
end

applyTM_MPO(MPS::Vector{<:Any}, MPOvec::Vector{<:MPOsparseT}, x::Vector{<:Any}; left::Bool = true) where {DT<:Number, D} = applyTM_MPO(MPS, MPS, MPOvec, x; left = left)

