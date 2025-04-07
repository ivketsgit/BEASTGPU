
module Strategies
export igd_Integrand_distilled, igd_Integrand_distilled_and_mul
using StaticArrays

struct Simplex{U,D,C,N,T}
    vertices::SVector{N,SVector{U,T}}
    tangents::SVector{D,SVector{U,T}}
    normals::SVector{C,SVector{U,T}}
    volume::T
end

function simplex_own(vertices::SVector{D1,P}) where {D1,P}
    U = length(P)
    D = D1 - 1
    C = U-D
    T = eltype(P)
    
    tangents = SVector{D, P}(map(i -> vertices[i] - vertices[end], 1:D))
    normals, volume = _normals_own(tangents, Val{C})
    return Simplex(vertices, tangents, normals, T(volume))
end

function _normals_own(tangents::SVector{2,SVector{2,T}}, ::Type{Val{0}}) where {T}

    t = tangents[1]
    s = tangents[2]
    v = (t[1]*s[2] - t[2]*s[1])/2

    P = SVector{2,T}
    SVector{0,P}(), v
end
function vertices_dom1()
    return [Float64[1.0,0.0],Float64[0.0,1.0],Float64[0.0,0.0]]
end
struct PulledBackIntegrand{I,C1,C2}
    igd::I
    chart1::C1
    chart2::C2
end
function pulledback_integrand_(igd,
    I, chart1,
    J, chart2)

    V_1 = vertices_dom1()
    V_2 = vertices_dom1()

    ichart1_ = simplex_own(SVector(V_1[I[1]], V_1[I[2]], V_1[I[3]]))
    ichart2_ = simplex_own(SVector(V_2[J[1]], V_2[J[2]], V_2[J[3]]))

    
    PulledBackIntegrand(igd, ichart1_, ichart2_)
end 


"""

own code 

"""

function barytocart_test(mani, u)
    r = copy(mani.vertices[length(mani.vertices)])
    r += mani.tangents[1] * u[1]
    r += mani.tangents[2] * u[2]
    return r
end

function neighborhood_own(p, bary)
    D = 2
    T = coordtype(p)
    P = SVector{D,T}
    cart = barytocart_test(p, Float64.(bary))
    return p, P(bary), cart, D
end

function igd_local_test_space(igd, p, bary, cart, jacobian, normal)
    u,v,w, = bary[1], bary[2], 1-bary[1]-bary[2]

    j = jacobian
    p = p
    a = p[1]-p[3]
    b = p[2]-p[3]
    σ = sign(normal[1] * (a[2]b[3]-a[3]b[2]) + normal[2] * (a[3]b[1] - a[1]b[3]) + normal[3] * (a[1]b[2] - a[2]b[1]))
    SVector(
        (value=u, curl=σ*(p[3]-p[2])/j),
        (value=v, curl=σ*(p[1]-p[3])/j),
        (value=w, curl=σ*(p[2]-p[1])/j))
end

function kernelvals_own(biop, cart_x, cart_y)
    α = biop.alpha
    γ = biop.gamma
    R = (cart_x[1] - cart_y[1])^2 + (cart_x[2] - cart_y[2])^2 + (cart_x[3] - cart_y[3])^2

    inv_R = 1/R

    inv_4pi = 1/(4pi)

    αgreen = α * exp(-γ*R) * inv_R * Float64(inv_4pi)

    return αgreen
end

#legacy
function igd_Integrand(funtion,u_,v_)
    u = barytocart_test(funtion.chart1,u_)
    v = barytocart_test(funtion.chart2,v_)
    igd = funtion.igd

    p_x, bary_x, cart_x, D_x = neighborhood_own(igd.test_chart,barytocart_test(funtion.chart1,u))
    p_y, bary_y, cart_y, D_y = neighborhood_own(igd.test_chart,barytocart_test(funtion.chart2,v))
    normal_x = normal(p_x)
    normal_y = normal(p_y)
    
    jacobian_x = p_x.volume * factorial(D_x)
    jacobian_y = p_y.volume * factorial(D_y)

    f = igd_local_test_space(igd, p_x, bary_x, cart_x, jacobian_x, normal_x)
    g = igd_local_test_space(igd, p_y, bary_y, cart_y, jacobian_y, normal_y)

    op = igd.operator
    
    green = kernelvals_own(op, cart_x, cart_y)

    N = 3
    M = 3
    result = Array{ComplexF64}(undef, (N, M))
    for m in 1:M
        for n in 1:N
            result[n, m] = green * f[n][1] * g[m][1]
        end
    end

    return jacobian_x * jacobian_y * result
end

function igd_Integrand__(funtion,u_,v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex)
    bary_x_1 = fc1v[3][1] + fc1t[1][1] * u_[1] + fc1t[2][1] * u_[2]
    bary_x_2 = fc1v[3][2] + fc1t[1][2] * u_[1] + fc1t[2][2] * u_[2]
    bary_y_1 = fc2v[3][1] + fc2t[1][1] * v_[1] + fc2t[2][1] * v_[2]
    bary_y_2 = fc2v[3][2] + fc2t[1][2] * v_[1] + fc2t[2][2] * v_[2]
    
    cart_x_1 = itecv[3][1] +  itect[1][1] * bary_x_1 + itect[2][1] * bary_x_2
    cart_x_2 = itecv[3][2] +  itect[1][2] * bary_x_1 + itect[2][2] * bary_x_2
    cart_x_3 = itecv[3][3] +  itect[1][3] * bary_x_1 + itect[2][3] * bary_x_2
    
    cart_y_1 = itrcv[3][1] +  itrct[1][1] * bary_y_1 + itrct[2][1] * bary_y_2
    cart_y_2 = itrcv[3][2] +  itrct[1][2] * bary_y_1 + itrct[2][2] * bary_y_2
    cart_y_3 = itrcv[3][3] +  itrct[1][3] * bary_y_1 + itrct[2][3] * bary_y_2

    jacobian_x = itecvo * 2
    jacobian_y = itrcvo * 2
    
    f = [bary_x_1, bary_x_2, 1-bary_x_1-bary_x_2]
    g = [bary_y_1, bary_y_2, 1-bary_y_1-bary_y_2]
    

    R = sqrt((cart_x_1 - cart_y_1)^2 + (cart_x_2 - cart_y_2)^2 + (cart_x_3 - cart_y_3)^2)
    inv_R = 1/R
    inv_4pi = 1/(4pi)
    green = exp(-γ*R) * inv_R * Float64(inv_4pi)

    # if mutex
    #     print(" R = ", R)
    #     mutex = false
    # end
    N = 3
    M = 3
    result = Array{ComplexF64}(undef, (N, M))
    for m in 1:M
        for n in 1:N
            result[n, m] = green * f[n] * g[m]
        end
    end
    
    return jacobian_x * jacobian_y * result
end


function StrategyCommonFace_cpu(f, η1, η2, η3, ξ)
    return (ξ^3) *
           ((η1)^2) *
           (η2) *
           (
            f((1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), (1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1)) +
            f((1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), (1 - ξ, ξ - ξ * η1 + ξ * η1 * η2)) +
            f((1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), (1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2))) +
            f((1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), (1 - ξ, ξ * η1 * (1 - η2 + η2 * η3))) +
            f((1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), (1 - ξ, ξ * η1 * (1 - η2))) +
            f((1 - ξ, ξ * η1 * (1 - η2)), (1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)))
           )
end

function StrategyCommonEdge_cpu(f, η1, η2, η3, ξ)

    ξη1 = ξ * η1
    η1η2 = η1 * η2
    η2η3 = η2 * η3
    η1η2η3 = η1η2 * η3

    return (ξ^3) * ((η1)^2) * igd_Integrand(f,(1 - ξ, ξη1 * η3), (1 - ξ * (1 - η1η2), ξη1 * (1 - η2))) +
           (ξ^3) *
           ((η1)^2) *
           (η2) *
           (
                igd_Integrand(f,(1 - ξ, ξη1), (1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3))) +
                igd_Integrand(f,(1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), (1 - ξ, ξη1 * η2η3)) +
                igd_Integrand(f,(1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), (1 - ξ, ξη1)) +
                igd_Integrand(f,(1 - ξ * (1 - η1η2η3), ξη1 * (1 - η2η3)), (1 - ξ, ξη1 * η2))
           )
end

function StrategyCommonVertex_cpu(f, η1, η2, η3, ξ)

    ξη1 = ξ * η1
    ξη2 = ξ * η2

    return (ξ^3) * η2 * (igd_Integrand(f,(1 - ξ, ξη1), (1 - ξη2, ξη2 * η3)) + igd_Integrand(f,(1 - ξη2, ξη2 * η3), (1 - ξ, ξη1)))
end


function StrategyCommonFace(f, η1, η2, η3, ξ, mutex)
    igd = f.igd
    fc1v = f.chart1.vertices
    fc1t = f.chart1.tangents
    itecv = igd.test_chart.vertices
    itect = igd.test_chart.tangents
    fc2v = f.chart1.vertices
    fc2t = f.chart1.tangents
    itrcv = igd.trial_chart.vertices
    itrct = igd.trial_chart.tangents
    itecvo = igd.test_chart.volume
    itrcvo = igd.trial_chart.volume
    γ = igd.operator.gamma
    return (ξ^3) *
           ((η1)^2) *
           (η2) *
           (
            igd_Integrand__(f,(1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), (1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
            igd_Integrand__(f,(1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), (1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
            igd_Integrand__(f,(1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), (1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
            igd_Integrand__(f,(1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), (1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
            igd_Integrand__(f,(1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), (1 - ξ, ξ * η1 * (1 - η2)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
            igd_Integrand__(f,(1 - ξ, ξ * η1 * (1 - η2)), (1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex)
           )
end


function StrategyCommonEdge(f, η1, η2, η3, ξ, mutex)
    igd = f.igd
    fc1v = f.chart1.vertices
    fc1t = f.chart1.tangents
    itecv = igd.test_chart.vertices
    itect = igd.test_chart.tangents
    fc2v = f.chart1.vertices
    fc2t = f.chart1.tangents
    itrcv = igd.trial_chart.vertices
    itrct = igd.trial_chart.tangents
    itecvo = igd.test_chart.volume
    itrcvo = igd.trial_chart.volume
    γ = igd.operator.gamma

    ξη1 = ξ * η1
    η1η2 = η1 * η2
    η2η3 = η2 * η3
    η1η2η3 = η1η2 * η3

    return (ξ^3) * ((η1)^2) * igd_Integrand__(f,(1 - ξ, ξη1 * η3), (1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
           (ξ^3) *
           ((η1)^2) *
           (η2) *
           (
                igd_Integrand__(f,(1 - ξ, ξη1), (1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
                igd_Integrand__(f,(1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), (1 - ξ, ξη1 * η2η3), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
                igd_Integrand__(f,(1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), (1 - ξ, ξη1), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) +
                igd_Integrand__(f,(1 - ξ * (1 - η1η2η3), ξη1 * (1 - η2η3)), (1 - ξ, ξη1 * η2), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex)
           )
end


function StrategyCommonVertex(f, η1, η2, η3, ξ, mutex)
    igd = f.igd
    fc1v = f.chart1.vertices
    fc1t = f.chart1.tangents
    itecv = igd.test_chart.vertices
    itect = igd.test_chart.tangents
    fc2v = f.chart1.vertices
    fc2t = f.chart1.tangents
    itrcv = igd.trial_chart.vertices
    itrct = igd.trial_chart.tangents
    itecvo = igd.test_chart.volume
    itrcvo = igd.trial_chart.volume
    γ = igd.operator.gamma

    ξη1 = ξ * η1
    ξη2 = ξ * η2

    return (ξ^3) * η2 * (igd_Integrand__(f,(1 - ξ, ξη1), (1 - ξη2, ξη2 * η3), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex) + 
                         igd_Integrand__(f,(1 - ξη2, ξη2 * η3), (1 - ξ, ξη1), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex))
end


const inv_4pi = 1/(4pi)        
function igd_Integrand_distilled(i, j, k, l, igd_Integrands, u_, v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ)
    
    # bary_x_1 = fc1v[3, 1] + fc1t[1, 1] * u_[1] + fc1t[2, 1] * u_[1]
    # bary_x_2 = fc1v[3, 2] + fc1t[1, 2] * u_[2] + fc1t[2, 2] * u_[2]

    
    # cart_x_1 = itecv[1, 3] + itect[1, 1] * bary_x_1 + itect[1, 2] * bary_x_2
    # cart_x_2 = itecv[2, 3] + itect[2, 1] * bary_x_1 + itect[2, 2] * bary_x_2
    # cart_x_3 = itecv[3, 3] + itect[3, 1] * bary_x_1 + itect[3, 2] * bary_x_2

    # bary_y_1 = fc2v[3, 1] + fc2t[1, 1] * v_[1] + fc2t[2, 1] * v_[1]
    # bary_y_2 = fc2v[3, 2] + fc2t[1, 2] * v_[2] + fc2t[2, 2] * v_[2]

    # cart_y_1 = itrcv[1, 3] + itrct[1, 1] * bary_y_1 + itrct[1, 2] * bary_y_2
    # cart_y_2 = itrcv[2, 3] + itrct[2, 1] * bary_y_1 + itrct[2, 2] * bary_y_2
    # cart_y_3 = itrcv[3, 3] + itrct[3, 1] * bary_y_1 + itrct[3, 2] * bary_y_2

    

    # R = sqrt((cart_x_1 - cart_y_1)^2 + (cart_x_2 - cart_y_2)^2 + (cart_x_3 - cart_y_3)^2)
    # green =  exp(-γ[1]*R) * inv_4pi / R

    # f_1 = bary_x_1
    # f_2 = bary_x_2
    # f_3 = 1 - bary_x_1 - bary_x_2
    # g_1 = bary_y_1
    # g_2 = bary_y_2
    # g_3 = 1 - bary_y_1 - bary_y_2

    # jacobian_x_mul_jacobian_y_mul_green = 4 * itecvo[1] * itrcvo[1] * green

    # jacobian_x_mul_jacobian_y_mul_green_f_1 = jacobian_x_mul_jacobian_y_mul_green * f_1
    # jacobian_x_mul_jacobian_y_mul_green_f_2 = jacobian_x_mul_jacobian_y_mul_green * f_2
    # jacobian_x_mul_jacobian_y_mul_green_f_3 = jacobian_x_mul_jacobian_y_mul_green * f_3

    # igd_Integrands[i, j, k, l, 1] += (jacobian_x_mul_jacobian_y_mul_green_f_1 * g_1)
    # igd_Integrands[i, j, k, l, 2] += (jacobian_x_mul_jacobian_y_mul_green_f_2 * g_1)
    # igd_Integrands[i, j, k, l, 3] += (jacobian_x_mul_jacobian_y_mul_green_f_3 * g_1)

    # igd_Integrands[i, j, k, l, 4] += (jacobian_x_mul_jacobian_y_mul_green_f_1 * g_2)
    # igd_Integrands[i, j, k, l, 5] += (jacobian_x_mul_jacobian_y_mul_green_f_2 * g_2)
    # igd_Integrands[i, j, k, l, 6] += (jacobian_x_mul_jacobian_y_mul_green_f_3 * g_2)

    # igd_Integrands[i, j, k, l, 7] += (jacobian_x_mul_jacobian_y_mul_green_f_1 * g_3)
    # igd_Integrands[i, j, k, l, 8] += (jacobian_x_mul_jacobian_y_mul_green_f_2 * g_3)
    # igd_Integrands[i, j, k, l, 9] += (jacobian_x_mul_jacobian_y_mul_green_f_3 * g_3)
end


"""
qps = rule.qps
    result = zeros((3,3))
    function CommonFace_(f, η1, η2, η3, ξ)
        return (ξ^3) *
               ((η1)^2) *
               (η2) *
               (
                   f((1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), (1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1)) +
                   f((1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), (1 - ξ, ξ - ξ * η1 + ξ * η1 * η2)) +
                   f((1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), (1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2))) +
                   f((1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), (1 - ξ, ξ * η1 * (1 - η2 + η2 * η3))) +
                   f((1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), (1 - ξ, ξ * η1 * (1 - η2))) +
                   f((1 - ξ, ξ * η1 * (1 - η2)), (1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)))
               )
    end
    function CommonFaceQuad(f, η1, η2, η3, ξ)

        ξη1 = ξ * η1 # auxiliary
    
        mξ   = (1 - ξ)# auxiliary
        mξη1 = (1 - ξη1)# auxiliary
    
        # only 4 different terms occur as argument:
        mξη3 = mξ * η3
        mξη3p = mξη3 + ξ
    
        mξη1η2  = mξη1 * η2
        mξη1η2p = mξη1η2 + ξη1
    
        return ξ *
               mξ *
               mξη1 *
               (
                   f((mξη3, mξη1η2), (mξη3p, mξη1η2p)) +
                   f((mξη1η2, mξη3), (mξη1η2p, mξη3p)) +
                   f((mξη3, mξη1η2p), (mξη3p, mξη1η2)) +
                   f((mξη1η2, mξη3p), (mξη1η2p, mξη3)) +
                   f((mξη3p, mξη1η2), (mξη3, mξη1η2p)) +
                   f((mξη1η2p, mξη3), (mξη1η2, mξη3p)) +
                   f((mξη3p, mξη1η2p), (mξη3, mξη1η2)) +
                   f((mξη1η2p, mξη3p), (mξη1η2, mξη3))
               )
    end
    function CommonEdge_copy(f, η1, η2, η3, ξ)
        ξη1 = ξ * η1
        η1η2 = η1 * η2
        η2η3 = η2 * η3
        η1η2η3 = η1η2 * η3
    
        return (ξ^3) * ((η1)^2) * f((1 - ξ, ξη1 * η3), (1 - ξ * (1 - η1η2), ξη1 * (1 - η2))) +
               (ξ^3) *
               ((η1)^2) *
               (η2) *
               (
                   f((1 - ξ, ξη1), (1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3))) +
                   f((1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), (1 - ξ, ξη1 * η2η3)) +
                   f((1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), (1 - ξ, ξη1)) +
                   f((1 - ξ * (1 - η1η2η3), ξη1 * (1 - η2η3)), (1 - ξ, ξη1 * η2))
               )
    end
    i = 1
    
    mutex = true
    if isa(rule, BEAST.SauterSchwabQuadrature.CommonFace)
        println("CommonFace")
        a = sum(w1 * w2 * w3 * w4 * CommonFace_(igdp, η1, η2, η3, ξ) for (η1, w1) in qps, (η2, w2) in qps, (η3, w3) in qps, (ξ, w4) in qps)
        for (η1, w1) in qps
            for (η2, w2) in qps
                for (η3, w3) in qps
                    for (ξ, w4) in qps
                        value = StrategyCommonFace_(igdp, η1, η2, η3, ξ, false)
                        result += w1 * w2 * w3 * w4 * value
        end end end end
    elseif isa(rule, BEAST.SauterSchwabQuadrature.CommonEdge)
        println("CommonEdge")
        a = sum(w1 * w2 * w3 * w4 * CommonEdge_copy(igdp, η1, η2, η3, ξ) for (η1, w1) in qps, (η2, w2) in qps, (η3, w3) in qps, (ξ, w4) in qps)
        for (η1, w1) in qps
            for (η2, w2) in qps
                for (η3, w3) in qps
                    for (ξ, w4) in qps
                        value = StrategyCommonEdge_cpu(igdp, η1, η2, η3, ξ, mutex)
                        mutex = false
                        result += w1 * w2 * w3 * w4 * value
        end end end end
    elseif isa(rule, BEAST.SauterSchwabQuadrature.CommonVertex)
        println("CommonVertex")
        a = sum(w1 * w2 * w3 * w4 * StrategyCommonVertex_cpu(igdp, η1, η2, η3, ξ) for (η1, w1) in qps, (η2, w2) in qps, (η3, w3) in qps, (ξ, w4) in qps)
        for (η1, w1) in qps
            for (η2, w2) in qps
                for (η3, w3) in qps
                    for (ξ, w4) in qps
                        value = StrategyCommonVertex_cpu(igdp, η1, η2, η3, ξ)
                        result += w1 * w2 * w3 * w4 * value
        end end end end
    else
        a = 0
        println("other") #mss PositiveDistance
    end

    diff = G - result
    norms = abs.(diff)
    @show maximum(norms)
    diff_2 = G - a
    norms_2 = abs.(diff_2)
    @show maximum(norms_2)

    # println("")
    # @show G
    # @show a
    # @show result
    # println("")
    # @show a
    # @show result
    
    gpu_result = Zzz.momintegrals_gpu_this!(op, test_local_space, trial_local_space, test_chart, trial_chart, out, rule)
    diff_3 = G - gpu_result
    norms_3 = abs.(diff_3)
    @show maximum(norms_3)

    # @show G
    # @show gpu_result
    # if isa(rule, BEAST.SauterSchwabQuadrature.CommonEdge)
    # throw("this")
    # end
"""

"""
function (igd::Integrand)(u,v)
    
    x = neighborhood(igd.test_chart,u)
    y = neighborhood(igd.trial_chart,v)
    #@show x
    # p_x, bary_x, cart_x, D_x = neighborhood_own(igd.test_chart, u)
    # p_y, bary_y, cart_y, D_y = neighborhood_own(igd.trial_chart, v)
    # normal_x = normal(p_x)
    # normal_y = normal(p_y)
    # @show normal_x, normal_y

    # @show p_x
    # @show x 
    # @show normal(x.patch)
    # println("")


    # p_x, bary_x, cart_x, D_x = neighborhood_own(igd.test_chart,barytocart_test(igd.chart1,u))
    # p_y, bary_y, cart_y, D_y = neighborhood_own(igd.test_chart,barytocart_test(igd.chart2,v))
    
    # jacobian_x = p_x.volume * 2 #factorial(D_x)
    # jacobian_y = p_y.volume * 2 #factorial(D_y)
    # @show jacobian_x, jacobian_y

    # f = igd_local_test_space(igd, p_x, bary_x, cart_x, jacobian_x, normal_x)
    # g = igd_local_test_space(igd, p_y, bary_y, cart_y, jacobian_y, normal_y)

    # @show f, g

    
    # @show @which igd.local_test_space(x)
    # f = igd.local_test_space(x)
    # g = igd.local_trial_space(y)

    # @show f, g

    # println(" ")

    # op = igd.operator
    
    # green = kernelvals_own(op, cart_x, cart_y)

    # N = 3
    # M = 3
    # result = Array{ComplexF64}(undef, (N, M))
    # for m in 1:M
    #     for n in 1:N
    #         result[n, m] =  green * f[n][1] * g[m][1] * jacobian_x * jacobian_y
    #     end
    # end

    # return jacobian_x * jacobian_y * result
    # @show f
    f = igd.local_test_space(x)
    # @show f
    #@show @which igd.local_test_space(x)
    # throw("jkdslmngqsl")
    g = igd.local_trial_space(y)
    # print(" Integrand ",result, " ", jacobian(x) * jacobian(y) * igd(x,y,f,g))
    # @show @which igd(x,y,f,g)

    return jacobian(x) * jacobian(y) * igd(x,y,f,g)
end
"""

"""
function (f::PulledBackIntegrand)(u,v)
    # In general I think a Jacobian determinant needs to be included. For Simplical and
    # Quadrilateral charts this is not needed because they are 1.
    # @show @which cartesian(f.chart1,u)
    # function barytocart(mani, u)
    #     r = last(mani.vertices)
    #     for i in 1 : dimension(mani)
    #         ti = mani.tangents[i]
    #         ui = u[i]
    #         #r += mani.tangents[i] * u[i]
    #         r += ti * ui
    #     end
    #     return r
    # end
    # @show @which f.igd(barytocart(f.chart1,u), barytocart(f.chart2,v))
    # @show f.igd(barytocart(f.chart1,u), barytocart(f.chart2,v))
    # @show f.igd(cartesian(f.chart1,u), cartesian(f.chart2,v))
    # return f.igd(barytocart(f.chart1,u), barytocart(f.chart2,v))
    
    f.igd(cartesian(f.chart1,u), cartesian(f.chart2,v))
end
"""

"""
function pulledback_integrand(igd,
    I, chart1,
    J, chart2)

    dom1 = domain(chart1)
    dom2 = domain(chart2)

    ichart1 = CompScienceMeshes.permute_vertices(dom1, I)
    function simplex_(vertices::SVector{D1,P}) where {D1,P}
        U = length(P)
        D = D1 - 1
        C = U-D
        T = eltype(P)
        xp1 =:(())
        for i in 1:D
            push!(xp1.args, :(vertices[\$i]-vertices[end]))
        end
        xp2 = :(SVector{\$D,P}(\$xp1))
        quote
            tangents = \$xp2
            normals, volume = _normals(tangents, Val{\$C})
            Simplex(vertices, tangents, normals, \$T(volume))
        end
    end

    function permute_vertices_(ch, I)
        V = vertices(ch)
        # @show simplex(SVector(V[I[1]], V[I[2]], V[I[3]]))
        # @show simplex_(SVector(V[I[1]], V[I[2]], V[I[3]]))
        return simplex_(SVector(V[I[1]], V[I[2]], V[I[3]]))
    end
    # @show permute_vertices_(dom1, I)
    # @show ichart1
     
    ichart2 = CompScienceMeshes.permute_vertices(dom2, J)

    PulledBackIntegrand(igd, ichart1, ichart2)
end 
"""

"""
struct Simplex_{U,D,C,N,T}
    vertices
    tangents
    normals
    volume
end

function simplex_own(vertices::SVector{D1,P}) where {D1,P}
    U = 2
    D = D1 - 1
    C = U-D
    T = eltype(P)
    
    tangents = SVector{D, P}(map(i -> vertices[i] - vertices[end], 1:D))
    normals, volume = _normals_own(tangents)
    return Simplex_(vertices, tangents, normals, T(volume))
end

function _normals_own(tangents) where {T}
    a = T
    t = tangents[1]
    s = tangents[2]
    v = (t[1]*s[2] - t[2]*s[1])/2

    P = SVector{2,Float64}
    SVector{0,P}(), v
end
function vertices_dom1()
    return [Float64[1.0,0.0],Float64[0.0,1.0],Float64[0.0,0.0]]
end
function pulledback_integrand_(igd,
    I, chart1,
    J, chart2)

    V_1 = vertices_dom1()
    V_2 = vertices_dom1()

    
    ichart1_ = simplex_own(SVector(V_1[I[1]], V_1[I[2]], V_1[I[3]]))
    ichart2_ = simplex_own(SVector(V_2[J[1]], V_2[J[2]], V_2[J[3]]))
    
    PulledBackIntegrand(igd, ichart1_, ichart2_)
end 
"""

"""
function StrategyCommonEdge_cpu(f, η1, η2, η3, ξ, mutex)
    igd = f.igd
    fc1v = f.chart1.vertices
    fc1t = f.chart1.tangents
    itecv = igd.test_chart.vertices
    itect = igd.test_chart.tangents
    fc2v = f.chart2.vertices
    fc2t = f.chart2.tangents
    itrcv = igd.trial_chart.vertices
    itrct = igd.trial_chart.tangents
    itecvo = igd.test_chart.volume
    itrcvo = igd.trial_chart.volume
    α = igd.operator.alpha
    γ = igd.operator.gamma

    ξη1 = ξ * η1
    η1η2 = η1 * η2
    η2η3 = η2 * η3
    η1η2η3 = η1η2 * η3


    return (ξ^3) * ((η1)^2) * igd_Integrand__(f,(1 - ξ, ξη1 * η3), (1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α,mutex) +
           (ξ^3) *
           ((η1)^2) *
           (η2) *
           (
                igd_Integrand__(f,(1 - ξ, ξη1), (1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α,mutex) +
                igd_Integrand__(f,(1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), (1 - ξ, ξη1 * η2η3), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α,mutex) +
                igd_Integrand__(f,(1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), (1 - ξ, ξη1), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α,mutex) +
                igd_Integrand__(f,(1 - ξ * (1 - η1η2η3), ξη1 * (1 - η2η3)), (1 - ξ, ξη1 * η2), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α,mutex)
           )
end

function StrategyCommonVertex_cpu(f, η1, η2, η3, ξ)
    igd = f.igd
    fc1v = f.chart1.vertices
    fc1t = f.chart1.tangents
    itecv = igd.test_chart.vertices
    itect = igd.test_chart.tangents
    fc2v = f.chart2.vertices
    fc2t = f.chart2.tangents
    itrcv = igd.trial_chart.vertices
    itrct = igd.trial_chart.tangents
    itecvo = igd.test_chart.volume
    itrcvo = igd.trial_chart.volume
    α = igd.operator.alpha
    γ = igd.operator.gamma

    ξη1 = ξ * η1
    ξη2 = ξ * η2

    return (ξ^3) * η2 * (igd_Integrand__(f,(1 - ξ, ξη1), (1 - ξη2, ξη2 * η3), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, false) + igd_Integrand__(f,(1 - ξη2, ξη2 * η3), (1 - ξ, ξη1), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, false))
end

# function igd_Integrand_distilled_(funtion, u_,v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ)

#     bary_x = copy(fc1v[2])
#     bary_x += fc1t[1] * u_[1] + fc1t[2] * u_[2]
#     cart_x = copy(itecv[3])
#     cart_x += itect[1] * bary_x[1] + itect[2] * bary_x[2]


    
#     bary_y = copy(fc2v[2])
#     bary_y += fc2t[1] * v_[1] + fc2t[2] * v_[2]
#     cart_y = copy(itrcv[3])
#     cart_y += itrct[1] * bary_y[1] + itrct[2] * bary_y[2]

#     jacobian_x = itecvo * 2
#     jacobian_y = itrcvo * 2

#     f = [bary_x[1], bary_x[2], 1-bary_x[1]-bary_x[2]]
#     g = [bary_y[1], bary_y[2], 1-bary_y[1]-bary_y[2]]


#     R = sqrt((cart_x[1] - cart_y[1])^2 + (cart_x[2] - cart_y[2])^2 + (cart_x[3] - cart_y[3])^2)
#     inv_R = 1/R
#     inv_4pi = 1/(4pi)
#     green = exp(-γ*R) * inv_R * inv_4pi

#     result = Array{ComplexF64}(undef, (3, 3))
#     for m in 1:3
#         for n in 1:3
#             result[n, m] = jacobian_x * jacobian_y * green * f[n] * g[m]
#         end
#     end
    
#     return result
# end



function igd_Integrand__(funtion,u_,v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mutex)
    
    bary_x_1 = fc1v[3][1] + fc1t[1][1] * u_[1] + fc1t[2][1] * u_[2]
    bary_x_2 = fc1v[3][2] + fc1t[1][2] * u_[1] + fc1t[2][2] * u_[2]

    bary_y_1 = fc2v[3][1] + fc2t[1][1] * v_[1] + fc2t[2][1] * v_[2]
    bary_y_2 = fc2v[3][2] + fc2t[1][2] * v_[1] + fc2t[2][2] * v_[2]


    cart_x_1 = itecv[3][1] +  itect[1][1] * bary_x_1 + itect[2][1] * bary_x_2
    cart_x_2 = itecv[3][2] +  itect[1][2] * bary_x_1 + itect[2][2] * bary_x_2
    cart_x_3 = itecv[3][3] +  itect[1][3] * bary_x_1 + itect[2][3] * bary_x_2
    
    cart_y_1 = itrcv[3][1] +  itrct[1][1] * bary_y_1 + itrct[2][1] * bary_y_2
    cart_y_2 = itrcv[3][2] +  itrct[1][2] * bary_y_1 + itrct[2][2] * bary_y_2
    cart_y_3 = itrcv[3][3] +  itrct[1][3] * bary_y_1 + itrct[2][3] * bary_y_2

    jacobian_x = itecvo * 2
    jacobian_y = itrcvo * 2

    f = [bary_x_1, bary_x_2, 1-bary_x_1-bary_x_2]
    g = [bary_y_1, bary_y_2, 1-bary_y_1-bary_y_2]
    

    R = sqrt((cart_x_1 - cart_y_1)^2 + (cart_x_2 - cart_y_2)^2 + (cart_x_3 - cart_y_3)^2)
    
    inv_R = 1/R
    inv_4pi = 1/(4pi)
    green = α * exp(-γ*R) * inv_R * Float64(inv_4pi)


    N = 3
    M = 3
    result = Array{ComplexF64}(undef, (N, M))
    for m in 1:M
        for n in 1:N
            result[n, m] = green * f[n] * g[m]
        end
    end
    
    return jacobian_x * jacobian_y * result
end

# function igd_Integrand_distilled(funtion,u_,v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ)
#     bary_x = copy(fc1v[2])
#     bary_x += fc1t[1] * u_[1] + fc1t[2] * u_[2]
    
#     cart_x = copy(itecv[3])
#     cart_x += itect[1] * bary_x[1] + itect[2] * bary_x[2]


#     bary_y = copy(fc2v[2])
#     bary_y += fc2t[1] * v_[1] + fc2t[2] * v_[2]

#     cart_y = copy(itrcv[3])
#     cart_y += itrct[1] * bary_y[1] + itrct[2] * bary_y[2]
    

#     jacobian_x = itecvo * 2
#     jacobian_y = itrcvo * 2
#     # @show itecvo, itrcvo

#     f = [bary_x[1], bary_x[2], 1-bary_x[1]-bary_x[2]]
#     g = [bary_y[1], bary_y[2], 1-bary_y[1]-bary_y[2]]
#     # @show f, g


#     R = sqrt((cart_x[1] - cart_y[1])^2 + (cart_x[2] - cart_y[2])^2 + (cart_x[3] - cart_y[3])^2)
#     inv_R = 1/R
#     inv_4pi = 1/(4pi) #Cosnt
#     green = exp(-γ*R) * inv_R * inv_4pi
#     # @show green
#     # @show jacobian_x * jacobian_y * green

#     result = Array{ComplexF64}(undef, (3, 3))
#     for m in 1:3
#         for n in 1:3
#             result[n, m] = green * f[n] * g[m]
#         end
#     end
    
#     return jacobian_x * jacobian_y * result
#     # return jacobian(x) * jacobian(y) * result
# end

function StrategyCommonFace_(f, η1, η2, η3, ξ, mutex)
    igd = f.igd
    fc1v = f.chart1.vertices
    fc1t = f.chart1.tangents
    itecv = igd.test_chart.vertices
    itect = igd.test_chart.tangents
    fc2v = f.chart2.vertices
    fc2t = f.chart2.tangents
    itrcv = igd.trial_chart.vertices
    itrct = igd.trial_chart.tangents
    itecvo = igd.test_chart.volume
    itrcvo = igd.trial_chart.volume
    α = igd.operator.alpha
    γ = igd.operator.gamma
    
    return (ξ^3) *
           ((η1)^2) *
           (η2) *
           (
            igd_Integrand__(f,(1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), (1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mutex) +
            igd_Integrand__(f,(1 - (ξ - ξ * η1 * η2 * η3), ξ - ξ * η1), (1 - ξ, ξ - ξ * η1 + ξ * η1 * η2), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mutex) +
            igd_Integrand__(f,(1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), (1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mutex) +
            igd_Integrand__(f,(1 - (ξ - ξ * η1 * η2), ξ * η1 * (1 - η2)), (1 - ξ, ξ * η1 * (1 - η2 + η2 * η3)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mutex) +
            igd_Integrand__(f,(1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), (1 - ξ, ξ * η1 * (1 - η2)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mutex) +
            igd_Integrand__(f,(1 - ξ, ξ * η1 * (1 - η2)), (1 - (ξ - ξ * η1 * η2 * η3), ξ * η1 * (1 - η2 * η3)), fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, α, mutex)
           )
end

# function StrategyCommonEdge(f, η1, η2, η3, ξ)

#     ξη1 = ξ * η1
#     η1η2 = η1 * η2
#     η2η3 = η2 * η3
#     η1η2η3 = η1η2 * η3

#     return (ξ^3) * ((η1)^2) * igd_Integrand(f,(1 - ξ, ξη1 * η3), (1 - ξ * (1 - η1η2), ξη1 * (1 - η2))) +
#            (ξ^3) *
#            ((η1)^2) *
#            (η2) *
#            (
#                 igd_Integrand(f,(1 - ξ, ξη1), (1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3))) +
#                 igd_Integrand(f,(1 - ξ * (1 - η1η2), ξη1 * (1 - η2)), (1 - ξ, ξη1 * η2η3)) +
#                 igd_Integrand(f,(1 - ξ * (1 - η1η2η3), ξη1 * η2 * (1 - η3)), (1 - ξ, ξη1)) +
#                 igd_Integrand(f,(1 - ξ * (1 - η1η2η3), ξη1 * (1 - η2η3)), (1 - ξ, ξη1 * η2))
#            )
# end

# function StrategyCommonVertex_(f, η1, η2, η3, ξ)

#     ξη1 = ξ * η1
#     ξη2 = ξ * η2

#     return (ξ^3) * η2 * (igd_Integrand(f,(1 - ξ, ξη1), (1 - ξη2, ξη2 * η3)) + igd_Integrand(f,(1 - ξη2, ξη2 * η3), (1 - ξ, ξη1)))
# end

# function igd_Integrand___(funtion,u_,v_, fc1v, fc1t, itecv, itect, fc2v, fc2t, itrcv, itrct, itecvo, itrcvo, γ, mutex)
#     bary_x_1 = fc1v[3][1] + fc1t[1][1] * u_[1] + fc1t[2][1] * u_[2]
#     bary_x_2 = fc1v[3][2] + fc1t[1][2] * u_[1] + fc1t[2][2] * u_[2]
#     bary_y_1 = fc2v[3][1] + fc2t[1][1] * v_[1] + fc2t[2][1] * v_[2]
#     bary_y_2 = fc2v[3][2] + fc2t[1][2] * v_[1] + fc2t[2][2] * v_[2]
    
#     cart_x_1 = itecv[3][1] +  itect[1][1] * bary_x_1 + itect[2][1] * bary_x_2
#     cart_x_2 = itecv[3][2] +  itect[1][2] * bary_x_1 + itect[2][2] * bary_x_2
#     cart_x_3 = itecv[3][3] +  itect[1][3] * bary_x_1 + itect[2][3] * bary_x_2
    
#     cart_y_1 = itrcv[3][1] +  itrct[1][1] * bary_y_1 + itrct[2][1] * bary_y_2
#     cart_y_2 = itrcv[3][2] +  itrct[1][2] * bary_y_1 + itrct[2][2] * bary_y_2
#     cart_y_3 = itrcv[3][3] +  itrct[1][3] * bary_y_1 + itrct[2][3] * bary_y_2

#     jacobian_x = itecvo * 2
#     jacobian_y = itrcvo * 2
    
#     f = [bary_x_1, bary_x_2, 1-bary_x_1-bary_x_2]
#     g = [bary_y_1, bary_y_2, 1-bary_y_1-bary_y_2]

#     R = sqrt((cart_x_1 - cart_y_1)^2 + (cart_x_2 - cart_y_2)^2 + (cart_x_3 - cart_y_3)^2)
#     inv_R = 1/R
#     inv_4pi = 1/(4pi)
#     green = exp(-γ*R) * inv_R * Float64(inv_4pi)

#     if mutex
#         print(" R = ", R)
#         mutex = false
#     end
#     N = 3
#     M = 3
#     result = Array{ComplexF64}(undef, (N, M))
#     for m in 1:M
#         for n in 1:N
#             result[n, m] = green * f[n] * g[m]
#         end
#     end
    
#     return jacobian_x * jacobian_y * result
# end



















# function kernelvals_own(biop, cart_x, cart_y)
#     γ = biop.gamma
#     R = (cart_x[1] - cart_y[1])^2 + (cart_x[2] - cart_y[2])^2 + (cart_x[3] - cart_y[3])^2

#     inv_R = 1/R

#     inv_4pi = 1/(4pi)

#     green = exp(-γ*R) * inv_R * Float64(inv_4pi)

#     return green
# end

# function igd_Integrand(funtion,u_,v_)
#     u = barytocart_test(funtion.chart1,u_)
#     v = barytocart_test(funtion.chart2,v_)
#     igd = funtion.igd

#     p_x, bary_x, cart_x, D_x = neighborhood_own(igd.test_chart,barytocart_test(funtion.chart1,u))
#     p_y, bary_y, cart_y, D_y = neighborhood_own(igd.test_chart,barytocart_test(funtion.chart2,v))
    
#     jacobian_x = p_x.volume * factorial(D_x)
#     jacobian_y = p_y.volume * factorial(D_y)

#     f = igd_local_test_space(igd, p_x, bary_x, cart_x, jacobian_x)
#     g = igd_local_test_space(igd, p_y, bary_y, cart_y, jacobian_y)

#     op = igd.operator
    
#     green = kernelvals_own(op, cart_x, cart_y)

#     N = 3
#     M = 3
#     result = Array{ComplexF64}(undef, (N, M))
#     for m in 1:M
#         for n in 1:N
#             result[n, m] = green * f[n][1] * g[m][1]
#         end
#     end

#     return jacobian_x * jacobian_y * result
# end
"""
end