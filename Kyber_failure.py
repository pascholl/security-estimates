import operator as op
from math import factorial as fac
from math import sqrt, log, exp
from collections import defaultdict
import sys
from proba_util import *

def p2_cyclotomic_final_error_distribution(ps, mod_switch=False):
    """ construct the final error distribution in our encryption scheme
    :param ps: parameter set (ParameterSet)
    """
    chis = build_centered_binomial_law(ps.ks)           # LWE error law for the key
    chie = build_centered_binomial_law(ps.ke_ct)        # LWE error law for the ciphertext
    chie_pk = build_centered_binomial_law(ps.ke)
    if mod_switch:
        Rk = build_mod_switching_error_law(ps.q, ps.rqk)    # Rounding error public key
        Rc = build_mod_switching_error_law(ps.q, ps.rqc)    # rounding error first ciphertext
    else:
        Rk = {0:1}
        Rc = {0:1}
    chiRs = law_convolution(chis, Rk)                   # LWE+Rounding error key
    chiRe = law_convolution(chie, Rc)                   # LWE + rounding error ciphertext

    B1 = law_product(chie_pk, chiRs)                       # (LWE+Rounding error) * LWE (as in a E*S product)
    B2 = law_product(chis, chiRe)

    C1 = iter_law_convolution(B1, ps.m * ps.d)
    C2 = iter_law_convolution(B2, ps.m * ps.d)

    C=law_convolution(C1, C2)

    if mod_switch:
        R2 = build_mod_switching_error_law(ps.q, ps.rq2)    # Rounding2 (in the ciphertext mask part)
    else:
        R2 = {0:1}
    F = law_convolution(R2, chie)                       # LWE+Rounding2 error
    D = law_convolution(C, F)                           # Final error
    return D


def p2_cyclotomic_error_probability(ps, offset=0):
    F = p2_cyclotomic_final_error_distribution(ps)
    proba = tail_probability(F, ps.q/4 - offset)
    return F, ps.d*proba

def log_failure(ps, offset=0):
    F, pr = p2_cyclotomic_error_probability(ps, offset)
    return log(pr + 2.**(-300))/log(2)

def compute_rd(P, Q, a=2):
    """ Compute RD_a(P || Q) using the definition """
    return sum(P[x]**a/Q[x]**(a-1) for x in P)**(1/(a-1))

def rd_cont_gaussians(offset, sigma0, sigma1, a=2):
    """ RD of two continuous Gaussians with parameters (offset, sigma0) and (0, sigma1) """
    sigma_a = sqrt((1-a)*sigma0**2 + a*sigma1**2)
    return exp(a * offset**2 / (2*sigma_a**2)) * (sigma_a / (sigma0**(1-a) * sigma1**a))**(1/(1-a))

def rd_rounded_gaussians(offset, sigma1, sigma2, a=2):
    """ RD of two rounded Gaussians with parameters (offset, sigma1) and (0, sigma2).
    Calculated via brute-force """
    D1 = build_rounded_gaussian_law(sigma1)
    D2 = build_rounded_gaussian_law(sigma2)
    D1 = law_add_constant(D1, offset)
    ret = compute_rd(D1, D2, a)
    return ret

def rd_uniform(offset, B1, B2, a=2):
    """ RD of Uniform([-B1, B1]) + offset and Uniform([-B2, B2]) """
    D1 = build_centered_uniform_law(B1)
    D2 = build_centered_uniform_law(B2)
    D1 = law_add_constant(D1, offset)
    ret = compute_rd(D1, D2, a)
    return ret

def kyber_rd_rounded_gaussian(ps, beta, l, a=2):
    """ Brute-force method """
    sigma = beta*ps.q
    offset = int(ps.q/4 - 6*sigma)
    D1 = build_rounded_gaussian_law(sigma)
    D2 = build_rounded_gaussian_law(sigma*offset)
    D1 = law_add_constant(D1, offset)
    cont_formula = 2*3.142*offset**2 / (beta*ps.q)**2
    nopi = offset**2 / (beta*ps.q)**2
    ret = compute_rd(D1, D2, a)
    print('pi: %.2f, nopi: %.2f, brute: %.2f' % (cont_formula, nopi, log(ret)))
    return ps.d * ret

def plot_rd(offset=300, step=100, rd_order=2):
    real_sigma = offset//2
    results = []

    for real_sigma in range(offset//2, offset//2 + 5000, 50):
        sigma_law = build_rounded_gaussian_law(real_sigma, tailcut=int(round(3.2*real_sigma)))
        D_real = law_add_constant(sigma_law, offset)
        rd_old = 2**40
        rd_new = 2**40-1
        # find ideal sigma with smallest 
        print('sigma', real_sigma)
        ideal_sigma = real_sigma
        while rd_new < rd_old:
            #try:
            D_ideal = build_rounded_gaussian_law(ideal_sigma, tailcut=10*ideal_sigma)
            tmp = rd_new
            rd_new = compute_rd(D_real, D_ideal, rd_order)
            rd_old = tmp
            #print('Ideal: %s, RD: %.2f' % (ideal_sigma, rd_new))
            #except:
            #    pass
            ideal_sigma += step
        results.append((real_sigma, rd_old))
    return results

def plot_rd_tailcut(offset=300):
    """ Try different tailcut pars """
    cut_start = 2.0
    cut_end = 4.0
    cut_step = 0.05
    rd_order = 2
    real_sigma = offset//2
    sigma_step = 20
    results = []
    tailcut = cut_start
    for i in range(int((cut_end - cut_start) / cut_step)):
        sigma_law = build_rounded_gaussian_law(real_sigma, tailcut=int(round(tailcut*real_sigma)))
        D_real = law_add_constant(sigma_law, offset)
        rd_old = 2**40
        rd_new = 2**40-1
        # find ideal sigma with smallest rd
        ideal_sigma = real_sigma
        while rd_new < rd_old:
            #try:
            D_ideal = build_rounded_gaussian_law(ideal_sigma, tailcut=10*ideal_sigma)
            tmp = rd_new
            rd_new = compute_rd(D_real, D_ideal, rd_order)
            rd_old = tmp
            #print('Ideal: %s, RD: %.2f' % (ideal_sigma, rd_new))
            #except:
            #    pass
            ideal_sigma += sigma_step
        results.append((tailcut, rd_old))
        tailcut += cut_step
    return results

def plot_rd_security(offset=300):
    base_sec = 256
    n = 256
    res = [0]*10
    for order in range(2,10):
        res[order] = plot_rd(offset, rd_order=order)
        for i,(sigma,rd) in enumerate(res[order]):
            sec = base_sec * (order-1)/order - n*log(rd, 2)
            res[order][i] = (sigma, sec)
    line_kwds = {'dpi': 150, 'thickness': 1}
    lines = [line(res[order], **line_kwds) for order in range(2,10)]
    return sum(lines), res

def plot_rd_uniform(offset=300, step=100, rd_order=2):
    real_sigma = offset//2
    results = []

    for real_k in range(offset//2, offset//2 + 20000, 50):
        unif_law = build_centered_uniform_law(real_k)
        D_real = law_add_constant(unif_law, offset)
        rd_old = 2**40
        rd_new = 2**40-1
        # find ideal sigma with smallest rd
        ideal_sigma = real_k
        while rd_new < rd_old:
            try:
                #D_ideal = build_rounded_gaussian_law(ideal_sigma, tailcut=10*ideal_sigma)
                D_ideal = build_centered_uniform_law(ideal_sigma)
                tmp = rd_new
                rd_new = compute_rd(D_real, D_ideal, rd_order)
                rd_old = tmp
                print('Ideal: %s, RD: %.2f' % (ideal_sigma, rd_new))
            except KeyError:
                pass
            ideal_sigma += 10
        sigma_from_k = sqrt(4*real_k**2 - 4*real_k)
        results.append((sigma_from_k, rd_old**256))
    return results

def kyber_find_sigma(ps, max_failure=2**(-40), l=1, sigma_start=230, sigma_step=5):
    eps = 1
    F = p2_cyclotomic_final_error_distribution(ps)
    sigma = sigma_start
    sd_cut = 3.1
    while eps > max_failure:
        sigma -= sigma_step
        sigma_law = build_rounded_gaussian_law(sigma, tailcut=int(round(sd_cut*sigma)))
        final_error_distribution = law_convolution(F, sigma_law)
        eps = ps.d * tail_probability(final_error_distribution, ps.q/4)
        max_flooding = sd_cut*sigma
        B = int(ps.q/4 - max_flooding)
        print('\tTrying sigma: %s, log failure: %.2f, max_B: %.2f' % (sigma, log(eps, 2), B))

    shift = int(ps.q/4 - max_flooding)

    print('Found sigma: %s, log failure: %.2f, max_ct_noise: %.2f' % (sigma, log(eps, 2), shift))
    sigma_law = build_rounded_gaussian_law(sigma, tailcut=int(round(sd_cut*sigma)))
    actual_ct_max = max(v for v in F if F[v] > max_failure)
    print('\t[ct max: %s, pr. 2^{%.3f}]' % (actual_ct_max, log(F[actual_ct_max],2)))
    D_real = law_add_constant(sigma_law, shift)
    # print('Real dist tails, left: %s, right: %s' % (-int(round(sd_cut*sigma)) + shift, int(round(sd_cut*sigma)) + shift))
    sigma_real = sigma
    rd_old = 2**40
    rd_new = 2**40-1
    rds = []
    rds_old = []
    while rd_new < rd_old:
        try:
            D_ideal = build_rounded_gaussian_law(sigma, tailcut=sd_cut*sigma_real + shift)
            tmp = rd_new
            rd_new = compute_rd(D_real, D_ideal)
            rd_old = tmp
            rds_old = rds
            print('Ideal sigma: %.2f, RD: %.3f' % (sigma, rd_new))
            rds = [compute_rd(D_real, D_ideal, i) for i in range(2,10)]
            #for i,rd in enumerate(rds):
            #    print('rd %s: %.2f, ' % (i+2, rds[i]), sep='')
            #print()
        except KeyboardInterrupt:
            pass
        sigma += sigma_step
        #    #print('Trying ideal sigma:', sigma)
        #    continue
    print('RD: ', rd_old)
    print('rds:', rds_old)

def renyi_divergence_log(ps, beta, l, debug=False):
    """ Renyi divergence between N(beta) and N(beta) + c,
    where N is a rounded normal distribution and c is the decryption error from ps, for l samples """
    beta_law = build_rounded_gaussian_law(beta*ps.q)
    F = p2_cyclotomic_final_error_distribution(ps)
    threshold_final_error_distribution = law_convolution(F, beta_law)
    eps = ps.d * tail_probability(threshold_final_error_distribution, ps.q/4)
    
    # 6 sigma gives naive tail bound 2^{-40}
    max_beta = 6*beta*ps.q
    F, naive_eps = p2_cyclotomic_error_probability(ps, offset=max_beta)
    if debug:
        print('eps:', log(eps + 2.**(-300),2))
        print('naive_eps:', log(naive_eps + 2.**(-300),2))
    if eps > 1:
        raise ValueError('Too large probability')

    B = int(ps.q/4 - max_beta)
    #ln_rd = 2*l*(log(1 + eps) - log(1 - eps))
    ln_rd = l * 2*3.142*ps.d*B**2 / (beta*ps.q)**2
    return ln_rd

if __name__ == '__main__':
    from Kyber import KyberParameterSet
    print('hi')
    #q = 3329
    #ps = KyberParameterSet(256, 4, 2, 2, q, 2**12, 2**12, 2**12)
    #rd = dict()
    #rd_calc = dict()

    #low = 50
    #high = 100
    #print('B:\t')
    #for B in range(low, high, 10):
    #    beta = B/q
    #    print(' %s ' % B, end='')
    #    rd[B] = renyi_divergence_log(ps, beta, 1)
    #    rd_calc[B] = kyber_rd_rounded_gaussian(ps, beta, 1)
    #print('\nrd1:\t', end='')
    #for B in range(low, high, 10):
    #    print(' %.2f ' % rd[B], end='')
    #print('\nrd2:\t ', end='')
    #for B in range(low, high, 10):
    #    print(' %.2f ' % rd_calc[B], end='')
